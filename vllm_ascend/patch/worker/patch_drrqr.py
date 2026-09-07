# SPDX-License-Identifier: Apache-2.0
"""Opt-in load-time monkeypatch, installed by each Ascend worker."""

from __future__ import annotations

import functools
import inspect
import logging

from vllm_ascend.drrqr import plan_from_config

logger = logging.getLogger(__name__)


def install_model_patch(model_cls, prepare_runtime=None) -> None:
    """Attach once per worker, preserving the no-plan baseline path."""
    if getattr(model_cls.__init__, "_ascend_drrqr_wrapper", False):
        return
    original_init = model_cls.__init__
    original_load = model_cls.load_weights
    if "vllm_config" not in inspect.signature(original_init).parameters:
        raise RuntimeError("DRRQR: Qwen model constructor interface changed")
    if "weights" not in inspect.signature(original_load).parameters:
        raise RuntimeError("DRRQR: Qwen weight loader interface changed")

    @functools.wraps(original_init)
    def initialize(self, *, vllm_config, prefix=""):
        plan = plan_from_config(vllm_config)
        if plan is not None and prepare_runtime is not None:
            prepare_runtime(plan)
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        self._ascend_drrqr_plan = plan
        if plan is not None:
            logger.info(
                "DRRQR configured before weight/cache loading: plan=%s Dk=%d->%d layers=%d",
                plan.digest,
                plan.old_head_k_dim,
                plan.target_head_k_dim,
                len(plan.keep_indices),
            )

    @functools.wraps(original_load)
    def load_weights(self, weights):
        plan = getattr(self, "_ascend_drrqr_plan", None)
        if plan is None:
            return original_load(self, weights)
        transformed = plan.transform_weights(weights)
        result = original_load(self, transformed)
        # A loader returning early skips coverage validation. Refuse it rather
        # than silently draining unconsumed inputs and claiming success.
        sentinel = object()
        if next(transformed, sentinel) is not sentinel:
            raise RuntimeError("DRRQR: original weight loader did not consume all weights")
        logger.info("DRRQR weight loading complete: plan=%s target_tensors=%d", plan.digest, 2 * len(plan.keep_indices))
        return result

    initialize._ascend_drrqr_wrapper = True
    model_cls.__init__ = initialize
    model_cls.load_weights = load_weights


def install_fused_shape_guard(gdn_module) -> None:
    """Route reduced shapes to Triton while preserving [N,Nv,Dv,Dk] state.

    Called only when constructing a DRRQR-enabled model. A service process owns
    one model; baseline rollback is a fresh service without ascend_drrqr config.
    The original availability probe tests 128x128, not the actual layer shape.
    No guard or transformation is added to the decode hot path.
    """
    cls = gdn_module.AscendGatedDeltaNetAttention
    original = cls._chunk_gated_delta_rule_fused
    if getattr(original, "_ascend_drrqr_wrapper", False):
        return

    @functools.wraps(original)
    def dispatch(q, k, v, g, beta, initial_state, cu_seqlens, scale):
        if q.shape[-1] == 128 and v.shape[-1] == 128:
            return original(q, k, v, g, beta, initial_state, cu_seqlens, scale)
        # The original caller passes this tensor directly from layer metadata.
        # Match by identity: no device synchronization or tensor comparison.
        # Rebuilding without prebuilt indices can underallocate variable-length
        # chunks (e.g. 65+65 tokens need four chunks, not ceil(130/64)=3).
        metadata = getattr(gdn_module.get_forward_context(), "attn_metadata", None)
        candidates = metadata.values() if isinstance(metadata, dict) else (metadata,)
        prebuilt_meta = None
        for candidate in candidates:
            if getattr(candidate, "prefill_query_start_loc", None) is not cu_seqlens:
                continue
            prefill = getattr(candidate, "non_spec_prefill_metadata", None)
            chunk = getattr(prefill, "chunk", None)
            if chunk is None:
                raise RuntimeError("DRRQR: matching prefill metadata has no chunk plan")
            if prebuilt_meta is not None and prebuilt_meta is not chunk:
                raise RuntimeError("DRRQR: ambiguous prefill chunk metadata")
            prebuilt_meta = chunk
        if prebuilt_meta is None:
            raise RuntimeError("DRRQR: no matching prebuilt prefill chunk metadata")
        output, final_state = gdn_module.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state.transpose(-1, -2).contiguous(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            scale=scale,
            prebuilt_meta=prebuilt_meta,
        )
        return output, final_state.transpose(-1, -2).contiguous()

    dispatch._ascend_drrqr_wrapper = True
    cls._chunk_gated_delta_rule_fused = staticmethod(dispatch)


def install() -> None:
    # Existing worker patch phase runs after the Qwen Ascend backend is
    # registered, including in every fresh multiprocessing-spawn worker.
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    from vllm_ascend.ops import gdn

    def prepare_runtime(plan):
        install_fused_shape_guard(gdn)

    install_model_patch(Qwen3_5Model, prepare_runtime=prepare_runtime)

# SPDX-License-Identifier: Apache-2.0
"""Validated load-time DRRQR transformation; no vLLM/Ascend imports here."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "ascend-drrqr-plan/v1"
OFFICIAL_COMMIT = "919d8667d951c385e08510bc1267c2e7049a4f56"
OFFICIAL_RRQR_SHA256 = "fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62"
WEIGHT_NAME = re.compile(r"^layers\.(\d+)\.linear_attn\.(in_proj_qkv|conv1d)\.weight$")
HASH = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"DRRQR: {name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DrrqrPlan:
    digest: str
    source_config_sha256: str
    source_index_sha256: str
    old_head_k_dim: int
    target_head_k_dim: int
    num_key_heads: int
    num_value_heads: int
    head_v_dim: int
    conv_kernel_dim: int
    hidden_size: int
    layer_types: tuple[str, ...]
    keep_indices: tuple[tuple[int, tuple[int, ...]], ...]

    @classmethod
    def load(cls, path: str, expected_sha256: str) -> DrrqrPlan:
        if not isinstance(expected_sha256, str) or not HASH.fullmatch(expected_sha256):
            raise ValueError("DRRQR: an explicit lowercase plan SHA256 is required")
        plan_path = Path(path)
        if not plan_path.is_absolute():
            raise ValueError("DRRQR: plan_path must be absolute and visible to every worker")
        raw = plan_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("DRRQR: plan hash mismatch")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("model_type") != "qwen3_5_text":
            raise ValueError("DRRQR: unsupported plan schema or model architecture")
        for name in ("source_config_sha256", "source_index_sha256"):
            if not isinstance(data.get(name), str) or not HASH.fullmatch(data[name]):
                raise ValueError(f"DRRQR: invalid {name}")
        provenance = data.get("provenance", {})
        if (
            not isinstance(provenance, dict)
            or provenance.get("method") != "DRRQR"
            or provenance.get("official_commit") != OFFICIAL_COMMIT
            or provenance.get("official_rrqr_sha256") != OFFICIAL_RRQR_SHA256
            or not isinstance(provenance.get("calibration_sha256"), str)
            or not HASH.fullmatch(provenance["calibration_sha256"])
        ):
            raise ValueError("DRRQR: missing pinned algorithm/calibration provenance")
        names = (
            "old_head_k_dim",
            "target_head_k_dim",
            "num_key_heads",
            "num_value_heads",
            "head_v_dim",
            "conv_kernel_dim",
            "hidden_size",
        )
        dims = {name: _positive_int(data.get(name), name) for name in names}
        old, new, heads = dims["old_head_k_dim"], dims["target_head_k_dim"], dims["num_key_heads"]
        if new >= old or dims["num_value_heads"] % heads:
            raise ValueError("DRRQR: invalid key reduction or value/key head ratio")
        layer_types = data.get("layer_types")
        if (
            not isinstance(layer_types, list)
            or not layer_types
            or any(not isinstance(kind, str) for kind in layer_types)
            or set(layer_types) != {"linear_attention", "full_attention"}
        ):
            raise ValueError("DRRQR: expected an explicit dense hybrid layer_types list")
        layer_ids = {i for i, kind in enumerate(layer_types) if kind == "linear_attention"}
        keeps = data.get("keep_indices")
        if not isinstance(keeps, dict) or set(keeps) != {str(i) for i in layer_ids}:
            raise ValueError("DRRQR: selection must cover exactly all linear-attention layers")
        validated = []
        for layer in sorted(layer_ids):
            keep = keeps[str(layer)]
            if not isinstance(keep, list) or len(keep) != heads * new:
                raise ValueError(f"DRRQR: incorrect selection width in layer {layer}")
            if any(type(i) is not int for i in keep) or len(set(keep)) != len(keep):
                raise ValueError(f"DRRQR: duplicate/non-integer indices in layer {layer}")
            for head in range(heads):
                head_keep = keep[head * new : (head + 1) * new]
                if any(i < head * old or i >= (head + 1) * old for i in head_keep):
                    raise ValueError(f"DRRQR: indices cross head boundaries in layer {layer}")
            validated.append((layer, tuple(keep)))
        return cls(
            digest=expected_sha256,
            source_config_sha256=data["source_config_sha256"],
            source_index_sha256=data["source_index_sha256"],
            layer_types=tuple(layer_types),
            keep_indices=tuple(validated),
            **dims,
        )

    def validate_source(self, model_path: str) -> None:
        root = Path(model_path)
        if not root.is_dir():
            raise ValueError("DRRQR: use the original local checkpoint, not a Hub id or derived checkpoint")
        for filename, digest in (
            ("config.json", self.source_config_sha256),
            ("model.safetensors.index.json", self.source_index_sha256),
        ):
            if file_sha256(root / filename) != digest:
                raise ValueError(f"DRRQR: original checkpoint {filename} hash mismatch")
        source = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.validate_text_config(source.get("text_config", source), reduced=False)
        if source.get("quantization_config"):
            raise ValueError("DRRQR: quantized checkpoints are unsupported")

    def validate_text_config(self, config: Any, *, reduced: bool) -> None:
        def get(name: str, default: Any = None) -> Any:
            return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)

        expected = {
            "model_type": "qwen3_5_text",
            "linear_key_head_dim": self.target_head_k_dim if reduced else self.old_head_k_dim,
            "linear_num_key_heads": self.num_key_heads,
            "linear_num_value_heads": self.num_value_heads,
            "linear_value_head_dim": self.head_v_dim,
            "linear_conv_kernel_dim": self.conv_kernel_dim,
            "hidden_size": self.hidden_size,
        }
        for name, value in expected.items():
            if get(name) != value:
                raise ValueError(f"DRRQR: {name} must be {value!r}, got {get(name)!r}")
        if tuple(get("layer_types", ())) != self.layer_types:
            raise ValueError("DRRQR: hybrid layer topology differs from the selection plan")
        if get("quantization_config") or get("num_experts", 0):
            raise ValueError("DRRQR: only unquantized dense Qwen GDN is supported")

    def validate_runtime(self, vllm_config: Any) -> None:
        model = vllm_config.model_config
        self.validate_source(model.model)
        self.validate_text_config(model.hf_text_config, reduced=True)
        if str(model.dtype) not in ("torch.bfloat16", "bfloat16") or model.quantization is not None:
            raise ValueError("DRRQR: initial runtime support is unquantized bfloat16 only")
        if getattr(vllm_config, "lora_config", None) or getattr(vllm_config, "speculative_config", None):
            raise ValueError("DRRQR: LoRA and speculative/MTP execution are not supported by this patch")
        parallel = vllm_config.parallel_config
        tp = parallel.tensor_parallel_size
        if (
            type(tp) is not int
            or tp < 1
            or self.num_key_heads % tp
            or self.num_value_heads % tp
            or parallel.pipeline_parallel_size != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != 1
            or getattr(parallel, "decode_context_parallel_size", 1) != 1
        ):
            raise ValueError("DRRQR: unsupported TP/PP/context-parallel configuration")
        load_format = getattr(getattr(vllm_config, "load_config", None), "load_format", "auto")
        if load_format not in ("auto", "safetensors"):
            raise ValueError("DRRQR: use original packed safetensors with auto/safetensors loader")

    def transform_weights(self, weights: Any):
        """Yield new Q/K tensors before TP sharding, retaining all other objects.

        No source tensor/file is changed. Consumers must exhaust the generator
        to verify the exact two-targets-per-linear-layer coverage.
        """
        import torch

        keeps = dict(self.keep_indices)
        required = {(layer, kind) for layer in keeps for kind in ("in_proj_qkv", "conv1d")}
        seen = set()
        qk_width = self.num_key_heads * self.old_head_k_dim
        packed_width = 2 * qk_width + self.num_value_heads * self.head_v_dim
        for name, weight in weights:
            match = WEIGHT_NAME.fullmatch(name)
            if not match:
                if ".linear_attn." in name and any(part in name for part in (".in_proj_qkv", ".conv1d.")):
                    raise ValueError(f"DRRQR: unsupported packed weight name {name}")
                yield name, weight
                continue
            layer, kind = int(match[1]), match[2]
            key = (layer, kind)
            if key not in required or key in seen:
                raise ValueError(f"DRRQR: unexpected or repeated target {name}")
            expected_shapes = (
                {(packed_width, self.hidden_size)}
                if kind == "in_proj_qkv"
                else {(packed_width, 1, self.conv_kernel_dim), (packed_width, self.conv_kernel_dim)}
            )
            if tuple(weight.shape) not in expected_shapes or not weight.is_floating_point():
                raise ValueError(f"DRRQR: unexpected source shape/dtype for {name}: {tuple(weight.shape)}")
            indices = torch.tensor(keeps[layer], dtype=torch.long, device=weight.device)
            reduced = torch.cat(
                (
                    weight[:qk_width].index_select(0, indices),
                    weight[qk_width : 2 * qk_width].index_select(0, indices),
                    weight[2 * qk_width :],
                ),
                dim=0,
            ).contiguous()
            seen.add(key)
            yield name, reduced
        if seen != required:
            raise ValueError(f"DRRQR: incomplete checkpoint target coverage: {sorted(required - seen)}")


def plan_from_config(vllm_config: Any) -> DrrqrPlan | None:
    metadata = getattr(vllm_config.model_config.hf_text_config, "ascend_drrqr", None)
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or set(metadata) != {"plan_path", "plan_sha256"}:
        raise ValueError("DRRQR: ascend_drrqr requires exactly plan_path and plan_sha256")
    plan = DrrqrPlan.load(metadata["plan_path"], metadata["plan_sha256"])
    plan.validate_runtime(vllm_config)
    return plan

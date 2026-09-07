# SPDX-License-Identifier: Apache-2.0
"""CPU tensor and upstream-interface contract tests for the DRRQR patch.

Run directly with Python or unittest; these tests intentionally do not load the
repository's NPU pytest conftest. Fake upstream classes exercise only the
constructor/loader contract, not vLLM or Ascend execution or performance.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import multiprocessing
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import the dependency-free implementation without running Ascend's package or
# worker initializers. This is deliberately not a stub for any tensor operation.
# Restore every temporary registration after import so collection does not
# contaminate subsequent tests that load the real Ascend package.
with mock.patch.dict(sys.modules):
    if "vllm_ascend" not in sys.modules:
        package = types.ModuleType("vllm_ascend")
        package.__path__ = [str(ROOT / "vllm_ascend")]
        sys.modules["vllm_ascend"] = package
    core = load_module("vllm_ascend.drrqr", ROOT / "vllm_ascend/drrqr.py")
    patch = load_module("_drrqr_worker_contract_test", ROOT / "vllm_ascend/patch/worker/patch_drrqr.py")


def legacy_prune_oracle():
    """Execute the existing converter's exact function without its SciPy CLI imports."""
    path = ROOT / "tools/state_reduction/build_qwen36_drrqr_checkpoint.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prune_packed")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["prune_packed"]


PRUNE_ORACLE = legacy_prune_oracle()


class PlanFixture:
    """Tiny, explicitly synthetic dense hybrid model; no model weights downloaded."""

    def __init__(self, root, target=64):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.text = {
            "model_type": "qwen3_5_text",
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "hidden_size": 8,
            "layer_types": ["linear_attention", "full_attention", "linear_attention"],
        }
        self.config_path = self.root / "config.json"
        self.index_path = self.root / "model.safetensors.index.json"
        self.config_path.write_text(json.dumps({"text_config": self.text}), encoding="utf-8")
        self.index_path.write_text(json.dumps({"weight_map": {}}), encoding="utf-8")
        self.data = {
            "schema": core.SCHEMA,
            "model_type": "qwen3_5_text",
            "source_config_sha256": core.file_sha256(self.config_path),
            "source_index_sha256": core.file_sha256(self.index_path),
            "old_head_k_dim": 128,
            "target_head_k_dim": target,
            "num_key_heads": 4,
            "num_value_heads": 8,
            "head_v_dim": 128,
            "conv_kernel_dim": 4,
            "hidden_size": 8,
            "layer_types": self.text["layer_types"],
            "provenance": {
                "method": "DRRQR",
                "official_commit": core.OFFICIAL_COMMIT,
                "official_rrqr_sha256": core.OFFICIAL_RRQR_SHA256,
                "calibration_sha256": hashlib.sha256(b"synthetic calibration fixture").hexdigest(),
            },
            "keep_indices": {},
        }
        for layer in (0, 2):
            generator = torch.Generator().manual_seed(710 + layer)
            # Noncontiguous, nonmonotonic selections expose accidental sorting,
            # head mixing, and Q/K misalignment.
            self.data["keep_indices"][str(layer)] = [
                head * 128 + index
                for head in range(4)
                for index in torch.randperm(128, generator=generator)[:target].tolist()
            ]
        self.path = self.root / "plan.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        self.digest = core.file_sha256(self.path)
        return self

    def load(self):
        return core.DrrqrPlan.load(str(self.path), self.digest)

    def config(self, *, enabled=True, tp=1, reduced=True):
        text = copy.deepcopy(self.text)
        if enabled:
            if reduced:
                text["linear_key_head_dim"] = self.data["target_head_k_dim"]
            text["ascend_drrqr"] = {
                "plan_path": str(self.path),
                "plan_sha256": self.digest,
            }
        return types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                model=str(self.root),
                hf_text_config=types.SimpleNamespace(**text),
                dtype=torch.bfloat16,
                quantization=None,
            ),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=tp,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            lora_config=None,
            speculative_config=None,
            load_config=types.SimpleNamespace(load_format="safetensors"),
        )

    def weights(self, *, convolution_2d=False):
        generator = torch.Generator().manual_seed(101)

        def tensor(shape):
            return torch.randint(-100, 100, shape, generator=generator).to(torch.bfloat16)

        packed = 2 * 4 * 128 + 8 * 128
        result = []
        for layer in (0, 2):
            prefix = f"layers.{layer}.linear_attn"
            result.extend(
                [
                    (prefix + ".in_proj_qkv.weight", tensor((packed, 8))),
                    (prefix + ".conv1d.weight", tensor((packed, 4) if convolution_2d else (packed, 1, 4))),
                    (prefix + ".in_proj_z.weight", tensor((8 * 128, 8))),
                    (prefix + ".in_proj_b.weight", tensor((8, 8))),
                    (prefix + ".in_proj_a.weight", tensor((8, 8))),
                    (prefix + ".A_log", tensor((8,))),
                    (prefix + ".dt_bias", tensor((8,))),
                    (prefix + ".out_proj.weight", tensor((8, 8 * 128))),
                ]
            )
        result.append(("layers.1.self_attn.q_proj.weight", tensor((128, 8))))
        result.append(("layers.1.mlp.gate_proj.weight", tensor((16, 8))))
        return result


def fake_model_class(*, consume_all=True):
    class Model:
        def __init__(self, *, vllm_config, prefix=""):
            self.config = vllm_config
            self.prefix = prefix
            self.constructed_head_dim = vllm_config.model_config.hf_text_config.linear_key_head_dim

        def load_weights(self, weights):
            self.input_object = weights
            self.loaded = list(weights) if consume_all else [next(iter(weights))]
            self.returned = {name for name, _ in self.loaded}
            return self.returned

    return Model


def fake_gdn_module():
    module = types.ModuleType("gdn_contract_fixture")
    module.original_calls = []
    module.triton_calls = []
    module.get_forward_context = mock.Mock(return_value=types.SimpleNamespace(attn_metadata=None))

    class Attention:
        @staticmethod
        def _chunk_gated_delta_rule_fused(q, k, v, g, beta, initial_state, cu_seqlens, scale):
            module.original_calls.append((q, k, v, g, beta, initial_state, cu_seqlens, scale))
            return v, initial_state

    def chunk(**kwargs):
        module.triton_calls.append(kwargs)
        return kwargs["v"], kwargs["initial_state"] + 1

    module.AscendGatedDeltaNetAttention = Attention
    module.chunk_gated_delta_rule = chunk
    return module


def spawned_install_and_load(model_root, result_queue):
    """A fresh spawn imports and installs the real patch against a fake upstream API."""
    try:
        fixture = PlanFixture(model_root, target=90)
        model_class = fake_model_class()
        qwen_module = types.ModuleType("vllm.model_executor.models.qwen3_5")
        qwen_module.Qwen3_5Model = model_class
        ops_module = types.ModuleType("vllm_ascend.ops")
        ops_module.gdn = fake_gdn_module()
        modules = {
            name: types.ModuleType(name) for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models")
        }
        modules.update({qwen_module.__name__: qwen_module, ops_module.__name__: ops_module})
        with mock.patch.dict(sys.modules, modules):
            patch.install()
            instance = model_class(vllm_config=fixture.config(tp=2))
            instance.load_weights(fixture.weights())
        result_queue.put(
            {
                "head_dim": instance.constructed_head_dim,
                "rows": instance.loaded[0][1].shape[0],
                "guard_installed": getattr(
                    ops_module.gdn.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused,
                    "_ascend_drrqr_wrapper",
                    False,
                ),
            }
        )
    except Exception as error:
        result_queue.put({"error": repr(error)})


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="drrqr-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.fixture = PlanFixture(self.tmp.name)

    def test_pinned_plan_and_source_hashes(self):
        fixture = self.fixture
        fixture.load().validate_source(str(fixture.root))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            core.DrrqrPlan.load(str(fixture.path), "0" * 64)
        for digest in (None, "A" * 64, "123"):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                core.DrrqrPlan.load(str(fixture.path), digest)
        with self.assertRaisesRegex(ValueError, "absolute"):
            core.DrrqrPlan.load("plan.json", fixture.digest)
        for path in (fixture.config_path, fixture.index_path):
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fixture.load().validate_source(str(fixture.root))
            path.write_bytes(original)

    def test_reject_invalid_topology_provenance_and_selection(self):
        fixture = self.fixture
        original = copy.deepcopy(fixture.data)
        mutations = {
            "missing layer": lambda d: d["keep_indices"].pop("2"),
            "full-attention selection": lambda d: d["keep_indices"].update({"1": d["keep_indices"]["0"]}),
            "duplicate channel": lambda d: d["keep_indices"]["0"].__setitem__(1, d["keep_indices"]["0"][0]),
            "cross head": lambda d: d["keep_indices"]["0"].__setitem__(0, 512),
            "noninteger channel": lambda d: d["keep_indices"]["0"].__setitem__(0, True),
            "wrong width": lambda d: d["keep_indices"]["0"].pop(),
            "nonhybrid": lambda d: d.update(layer_types=["linear_attention"] * 3),
            "unknown architecture": lambda d: d.update(model_type="unverified_qwen3_8"),
            "no reduction": lambda d: d.update(target_head_k_dim=128),
            "zero width": lambda d: d.update(target_head_k_dim=0),
            "bad head ratio": lambda d: d.update(num_value_heads=7),
            "unpinned code": lambda d: d["provenance"].update(official_commit="unknown"),
            "missing calibration": lambda d: d["provenance"].pop("calibration_sha256"),
            "nonobject provenance": lambda d: d.update(provenance=[]),
            "nonstring layer type": lambda d: d.update(layer_types=["linear_attention", [], "full_attention"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                fixture.data = copy.deepcopy(original)
                mutate(fixture.data)
                fixture.save()
                with self.assertRaises(ValueError):
                    fixture.load()

    def test_reject_nonobject_plan_document(self):
        for document in ([], None, "not an object", 42):
            with self.subTest(document=document):
                self.fixture.data = document
                self.fixture.save()
                with self.assertRaises(ValueError):
                    self.fixture.load()

    def test_reject_unsupported_runtime_contract(self):
        mutations = {
            "unreduced config": lambda c: setattr(c.model_config.hf_text_config, "linear_key_head_dim", 128),
            "FP16": lambda c: setattr(c.model_config, "dtype", torch.float16),
            "quantization": lambda c: setattr(c.model_config, "quantization", "awq"),
            "lora": lambda c: setattr(c, "lora_config", object()),
            "speculation": lambda c: setattr(c, "speculative_config", object()),
            "PP": lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
            "bad TP": lambda c: setattr(c.parallel_config, "tensor_parallel_size", 3),
            "prefill CP": lambda c: setattr(c.parallel_config, "prefill_context_parallel_size", 2),
            "decode CP": lambda c: setattr(c.parallel_config, "decode_context_parallel_size", 2),
            "loader": lambda c: setattr(c.load_config, "load_format", "pt"),
            "MoE": lambda c: setattr(c.model_config.hf_text_config, "num_experts", 16),
            "metadata keys": lambda c: c.model_config.hf_text_config.ascend_drrqr.update(extra=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                config = self.fixture.config()
                mutate(config)
                with self.assertRaises(ValueError):
                    core.plan_from_config(config)
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                self.assertEqual(core.plan_from_config(self.fixture.config(tp=tp)).target_head_k_dim, 64)


class TensorTransformTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="drrqr-tensors-")
        self.addCleanup(self.tmp.cleanup)

    def test_three_dimensions_match_converter_preserve_other_weights_and_tp_heads(self):
        for target in (64, 90, 102):
            for conv2d in (False, True):
                with self.subTest(target=target, convolution_2d=conv2d):
                    fixture = PlanFixture(Path(self.tmp.name) / f"dk{target}-{conv2d}", target)
                    plan = fixture.load()
                    originals = fixture.weights(convolution_2d=conv2d)
                    snapshots = {name: value.clone() for name, value in originals}
                    converted = dict(plan.transform_weights(iter(originals)))
                    for name, original in originals:
                        self.assertTrue(torch.equal(original, snapshots[name]), name)
                        match = core.WEIGHT_NAME.fullmatch(name)
                        if not match:
                            self.assertIs(converted[name], original, name)
                            continue
                        keep = torch.tensor(fixture.data["keep_indices"][match[1]])
                        actual = converted[name]
                        oracle = PRUNE_ORACLE(original, 512, keep)
                        self.assertTrue(torch.equal(actual, oracle), name)
                        self.assertTrue(actual.is_contiguous())
                        self.assertNotEqual(actual.data_ptr(), original.data_ptr())
                        self.assertTrue(torch.equal(actual[2 * 4 * target :], original[1024:]), "V changed")
                        # Reference the Q/K/V segments independently, then select
                        # local heads. This verifies transform-before-TP ordering;
                        # it does not claim to run vLLM's real TP weight loader.
                        for tp in (1, 2, 4):
                            for rank in range(tp):
                                h0, h1 = rank * 4 // tp, (rank + 1) * 4 // tp
                                local_keep = keep[h0 * target : h1 * target]
                                v0, v1 = rank * 1024 // tp, (rank + 1) * 1024 // tp
                                expected = torch.cat(
                                    (
                                        original[:512].index_select(0, local_keep),
                                        original[512:1024].index_select(0, local_keep),
                                        original[1024 + v0 : 1024 + v1],
                                    )
                                )
                                observed = torch.cat(
                                    (
                                        actual[h0 * target : h1 * target],
                                        actual[4 * target + h0 * target : 4 * target + h1 * target],
                                        actual[8 * target + v0 : 8 * target + v1],
                                    )
                                )
                                self.assertTrue(torch.equal(observed, expected), (name, tp, rank))

    def test_reject_missing_duplicate_unexpected_and_malformed_targets(self):
        fixture = PlanFixture(self.tmp.name)
        plan = fixture.load()
        weights = fixture.weights()
        bad_cases = {
            "missing": weights[1:],
            "duplicate": weights + [weights[0]],
            "full layer target": weights + [("layers.1.linear_attn.in_proj_qkv.weight", weights[0][1])],
            "prefixed target": [("model." + weights[0][0], weights[0][1])] + weights[1:],
            "qkv bias": weights + [("layers.0.linear_attn.in_proj_qkv.bias", torch.zeros(2048))],
            "bad shape": [(weights[0][0], weights[0][1][:-1])] + weights[1:],
            "integer tensor": [(weights[0][0], weights[0][1].to(torch.int32))] + weights[1:],
        }
        for name, values in bad_cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                list(plan.transform_weights(iter(values)))


class MonkeypatchContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="drrqr-monkeypatch-")
        self.addCleanup(self.tmp.cleanup)
        self.fixture = PlanFixture(self.tmp.name)

    def test_baseline_identity_no_preparation_and_idempotent_installation(self):
        model_class = fake_model_class()
        preparation = mock.Mock()
        patch.install_model_patch(model_class, prepare_runtime=preparation)
        init, loader = model_class.__init__, model_class.load_weights
        patch.install_model_patch(model_class, prepare_runtime=preparation)
        self.assertIs(model_class.__init__, init)
        self.assertIs(model_class.load_weights, loader)
        config = self.fixture.config(enabled=False)
        instance = model_class(vllm_config=config, prefix="baseline")
        weights = self.fixture.weights()
        returned = instance.load_weights(weights)
        preparation.assert_not_called()
        self.assertIs(instance.config, config)
        self.assertEqual(instance.constructed_head_dim, 128)
        self.assertIs(instance.input_object, weights)
        self.assertIs(returned, instance.returned)
        for (_, before), (_, after) in zip(weights, instance.loaded):
            self.assertIs(before, after)

    def test_reduced_config_reaches_constructor_and_loader_transforms(self):
        events = []
        model_class = fake_model_class()
        original_init = model_class.__init__

        def observe_init(self, *, vllm_config, prefix=""):
            events.append(("construct", vllm_config.model_config.hf_text_config.linear_key_head_dim))
            original_init(self, vllm_config=vllm_config, prefix=prefix)

        model_class.__init__ = observe_init
        patch.install_model_patch(
            model_class, prepare_runtime=lambda plan: events.append(("prepare", plan.target_head_k_dim))
        )
        instance = model_class(vllm_config=self.fixture.config(tp=4))
        self.assertEqual(events, [("prepare", 64), ("construct", 64)])
        weights = self.fixture.weights()
        returned = instance.load_weights(iter(weights))
        self.assertIs(returned, instance.returned)
        self.assertEqual(instance.loaded[0][1].shape, (1536, 8))
        self.assertEqual(len(instance.loaded), len(weights))

    def test_unreduced_driver_config_fails_before_construction(self):
        model_class = fake_model_class()
        preparation = mock.Mock()
        patch.install_model_patch(model_class, prepare_runtime=preparation)
        with self.assertRaisesRegex(ValueError, "linear_key_head_dim"):
            model_class(vllm_config=self.fixture.config(reduced=False))
        preparation.assert_not_called()

    def test_changed_upstream_signatures_are_rejected(self):
        for kind in ("constructor", "loader"):
            with self.subTest(interface=kind):
                model_class = fake_model_class()
                if kind == "constructor":
                    model_class.__init__ = lambda self, config: None
                else:
                    model_class.load_weights = lambda self, tensors: None
                before = model_class.__init__
                with self.assertRaisesRegex(RuntimeError, "interface changed"):
                    patch.install_model_patch(model_class)
                self.assertIs(model_class.__init__, before)

    def test_loader_rejects_short_consumption_and_incomplete_coverage(self):
        short_model = fake_model_class(consume_all=False)
        patch.install_model_patch(short_model)
        instance = short_model(vllm_config=self.fixture.config())
        with self.assertRaisesRegex(RuntimeError, "did not consume all weights"):
            instance.load_weights(iter(self.fixture.weights()))
        model_class = fake_model_class()
        patch.install_model_patch(model_class)
        instance = model_class(vllm_config=self.fixture.config())
        weights = self.fixture.weights()
        for bad in (weights[1:], weights + [weights[0]]):
            with self.subTest(count=len(bad)), self.assertRaises(ValueError):
                instance.load_weights(iter(bad))

    def test_spawned_worker_installs_and_transforms(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        self.addCleanup(queue.close)
        process = context.Process(target=spawned_install_and_load, args=(str(Path(self.tmp.name) / "spawn"), queue))
        process.start()
        process.join(timeout=45)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("spawned CPU contract test exceeded 45 seconds")
        self.assertEqual(process.exitcode, 0)
        result = queue.get(timeout=5)
        self.assertNotIn("error", result, result)
        self.assertEqual(result, {"head_dim": 90, "rows": 1744, "guard_installed": True})


class FusedShapeGuardTests(unittest.TestCase):
    def test_original_shape_preserves_fused_path_and_install_is_idempotent(self):
        module = fake_gdn_module()
        patch.install_fused_shape_guard(module)
        dispatch = module.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused
        patch.install_fused_shape_guard(module)
        self.assertIs(dispatch, module.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused)
        q = torch.zeros(1, 3, 4, 128, dtype=torch.bfloat16)
        v = torch.ones(1, 3, 8, 128, dtype=torch.bfloat16)
        state = torch.zeros(1, 8, 128, 128, dtype=torch.bfloat16)
        output, final = dispatch(q, q, v, None, None, state, None, 0.125)
        self.assertIs(output, v)
        self.assertIs(final, state)
        self.assertEqual(len(module.original_calls), 1)
        self.assertEqual(module.triton_calls, [])
        module.get_forward_context.assert_not_called()

    def test_reduced_shapes_use_triton_with_correct_state_layout_scale_and_normalization(self):
        for target in (64, 90, 102):
            with self.subTest(target=target):
                module = fake_gdn_module()
                patch.install_fused_shape_guard(module)
                q = torch.zeros(1, 3, 4, target, dtype=torch.bfloat16)
                k = torch.ones_like(q)
                v = torch.ones(1, 3, 8, 128, dtype=torch.bfloat16)
                state = (torch.arange(2 * 8 * 128 * target) % 31).reshape(2, 8, 128, target).to(torch.bfloat16)
                snapshot = state.clone()
                g, beta, lengths = torch.zeros(1, 3, 8), torch.ones(1, 3, 8), torch.tensor([0, 1, 3])
                metadata = object()
                candidate = types.SimpleNamespace(
                    prefill_query_start_loc=lengths,
                    non_spec_prefill_metadata=types.SimpleNamespace(chunk=metadata),
                )
                module.get_forward_context.return_value = types.SimpleNamespace(
                    attn_metadata={"layers.0": candidate, "layers.2": candidate}
                )
                scale = target**-0.5
                output, final = module.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                    q, k, v, g, beta, state, lengths, scale
                )
                self.assertEqual(module.original_calls, [])
                call = module.triton_calls[0]
                for name, original in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("cu_seqlens", lengths)):
                    self.assertIs(call[name], original)
                self.assertEqual(call["scale"], scale)
                self.assertIs(call["prebuilt_meta"], metadata)
                self.assertTrue(call["use_qk_l2norm_in_kernel"])
                self.assertTrue(call["output_final_state"])
                self.assertFalse(call["head_first"])
                self.assertEqual(call["initial_state"].shape, (2, 8, target, 128))
                self.assertTrue(call["initial_state"].is_contiguous())
                self.assertTrue(torch.equal(call["initial_state"], snapshot.transpose(-1, -2)))
                self.assertIs(output, v)
                self.assertTrue(torch.equal(final, snapshot + 1))
                self.assertTrue(final.is_contiguous())
                self.assertTrue(torch.equal(state, snapshot))

    def test_missing_mismatched_or_ambiguous_metadata_fails_before_kernel(self):
        q = torch.zeros(1, 3, 4, 64, dtype=torch.bfloat16)
        v = torch.ones(1, 3, 8, 128, dtype=torch.bfloat16)
        state = torch.zeros(2, 8, 128, 64, dtype=torch.bfloat16)
        lengths = torch.tensor([0, 1, 3])

        def candidate(location=lengths, chunk=None):
            return types.SimpleNamespace(
                prefill_query_start_loc=location,
                non_spec_prefill_metadata=types.SimpleNamespace(chunk=chunk),
            )

        contexts = {
            "no context": None,
            "no attention metadata": types.SimpleNamespace(attn_metadata=None),
            "empty attention metadata": types.SimpleNamespace(attn_metadata={}),
            "equal-valued different boundaries object": types.SimpleNamespace(
                attn_metadata={"layer": candidate(lengths.clone(), object())}
            ),
            "missing prefill metadata": types.SimpleNamespace(
                attn_metadata={"layer": types.SimpleNamespace(prefill_query_start_loc=lengths)}
            ),
            "missing chunk": types.SimpleNamespace(attn_metadata={"layer": candidate()}),
            "different chunk objects for same boundaries": types.SimpleNamespace(
                attn_metadata={"layer0": candidate(chunk=object()), "layer2": candidate(chunk=object())}
            ),
        }
        for name, context in contexts.items():
            with self.subTest(case=name):
                module = fake_gdn_module()
                module.get_forward_context.return_value = context
                patch.install_fused_shape_guard(module)
                with self.assertRaises(RuntimeError):
                    module.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                        q, q, v, None, None, state, lengths, 0.125
                    )
                self.assertEqual(module.original_calls, [])
                self.assertEqual(module.triton_calls, [])

    def test_65_plus_65_requests_forward_four_prebuilt_chunks(self):
        """CPU forwarding regression only: two 65-token requests need four chunks.

        Treating 130 packed tokens as one sequence would yield three 64-token
        chunks. Preserve the supplied per-request metadata and tensor identity;
        this test does not execute the real Triton kernel or claim NPU validation.
        """
        lengths = torch.tensor([0, 65, 130], dtype=torch.int32)
        chunk = types.SimpleNamespace(
            chunk_indices_chunk64=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int32),
            chunk_indices_chunk64_host=(0, 0, 0, 1, 1, 0, 1, 1),
            chunk_offsets_chunk64=torch.tensor([0, 2, 4], dtype=torch.int32),
            final_chunk_indices_chunk64=torch.tensor([1, 3], dtype=torch.int32),
        )
        for target in (64, 90, 102):
            with self.subTest(target=target):
                module = fake_gdn_module()
                candidate = types.SimpleNamespace(
                    prefill_query_start_loc=lengths,
                    non_spec_prefill_metadata=types.SimpleNamespace(chunk=chunk),
                )
                # An unrelated attention layer must not replace the matching
                # metadata or count as an ambiguous candidate.
                module.get_forward_context.return_value = types.SimpleNamespace(
                    attn_metadata={"full_attention": object(), "linear_attention": candidate}
                )
                patch.install_fused_shape_guard(module)
                q = torch.zeros(1, 130, 4, target, dtype=torch.bfloat16)
                v = torch.ones(1, 130, 8, 128, dtype=torch.bfloat16)
                state = torch.zeros(2, 8, 128, target, dtype=torch.bfloat16)
                module.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                    q, q, v, torch.zeros(1, 130, 8), torch.ones(1, 130, 8), state, lengths, target**-0.5
                )
                call = module.triton_calls[0]
                self.assertIs(call["cu_seqlens"], lengths)
                self.assertIs(call["prebuilt_meta"], chunk)
                self.assertEqual(call["prebuilt_meta"].chunk_indices_chunk64.tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])
                self.assertEqual(call["prebuilt_meta"].chunk_offsets_chunk64.tolist(), [0, 2, 4])
                self.assertEqual(len(call["prebuilt_meta"].chunk_indices_chunk64), 4)
                self.assertNotEqual(len(call["prebuilt_meta"].chunk_indices_chunk64), (130 + 63) // 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)

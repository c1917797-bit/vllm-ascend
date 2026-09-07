# SPDX-License-Identifier: Apache-2.0
"""Run independently: python -I -B tests/ut/tools/test_prepare_drrqr_plan.py."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "state_reduction"))
subject = importlib.import_module("prepare_drrqr_plan")


class PreparePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.capture = self.root / "capture"
        self.capture.mkdir()
        self.calibration = self.root / "calibration.jsonl"
        self.calibration.write_text('{"text":"independent calibration fixture"}\n', encoding="utf-8")
        self.official = self.root / "rrqr.py"
        self.official.write_text("# hash fixture; production requires official pinned bytes\n", encoding="utf-8")
        self.official_hash = hashlib.sha256(self.official.read_bytes()).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(subject.builder, "OFFICIAL_RRQR_SHA256", self.official_hash).start()
        self.output = self.root / "plan.json"
        self.write_source()

    def write_source(self, *, old_dim=4, nested=True, dtype=torch.bfloat16):
        text_config = {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "linear_key_head_dim": old_dim,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 3,
            "linear_conv_kernel_dim": 4,
            "hidden_size": 5,
            "num_hidden_layers": 3,
            "layer_types": ["linear_attention", "full_attention", "linear_attention"],
        }
        self.config = {"model_type": "qwen3_5", "text_config": text_config} if nested else text_config
        self.text_config = text_config
        tensors = {}
        width = 4 * old_dim + 12
        for layer in (0, 2):
            prefix = f"model.language_model.layers.{layer}.linear_attn"
            tensors[f"{prefix}.in_proj_qkv.weight"] = torch.arange(width * 5).reshape(width, 5).to(dtype)
            tensors[f"{prefix}.conv1d.weight"] = torch.arange(width * 4).reshape(width, 1, 4).to(dtype)
        tensors["model.language_model.layers.1.self_attn.q_proj.weight"] = torch.ones(5, 5, dtype=dtype)
        save_file(tensors, self.source / "model-00001-of-00001.safetensors")
        self.index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}
        self.save_metadata()

    def save_metadata(self):
        (self.source / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.source / "model.safetensors.index.json").write_text(json.dumps(self.index), encoding="utf-8")

    def selection(self, dim=2):
        old_dim = self.text_config["linear_key_head_dim"]
        values = list(range(dim)) + list(range(old_dim, old_dim + dim))
        return {layer: torch.tensor(values) for layer in (0, 2)}, {
            "schema": "selection-test",
            "layers": {"0": {}, "2": {}},
        }

    def run_prepare(self, *, dim=2, **kwargs):
        return subject.prepare_plan(
            self.source,
            self.capture,
            self.calibration,
            self.official,
            dim,
            self.output,
            tp_size=2,
            captures_per_rank=1,
            **kwargs,
        )

    def test_nested_plan_and_overrides_preserve_source(self):
        before = {path.name: subject.sha256(path) for path in self.source.iterdir()}
        with patch.object(subject.builder, "load_keep_maps", return_value=self.selection()) as choose:
            result = self.run_prepare()
        plan = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(plan["schema"], "ascend-drrqr-plan/v1")
        self.assertEqual(plan["model_type"], "qwen3_5_text")
        self.assertEqual(plan["keep_indices"], {"0": [0, 1, 4, 5], "2": [0, 1, 4, 5]})
        self.assertEqual(plan["provenance"]["selection"], self.selection()[1])
        self.assertEqual(result["plan_sha256"], subject.sha256(self.output))
        self.assertEqual(result["hf_overrides"]["text_config"]["linear_key_head_dim"], 2)
        self.assertEqual(result["hf_overrides"]["text_config"]["ascend_drrqr"]["plan_path"], str(self.output.resolve()))
        self.assertEqual(before, {path.name: subject.sha256(path) for path in self.source.iterdir()})
        self.assertEqual(choose.call_args.kwargs["expected_layer_ids"], [0, 2])

    def test_text_only_overrides_are_flat(self):
        self.write_source(nested=False)
        with patch.object(subject.builder, "load_keep_maps", return_value=self.selection()):
            result = self.run_prepare()
        self.assertNotIn("text_config", result["hf_overrides"])
        self.assertEqual(result["hf_overrides"]["linear_key_head_dim"], 2)

    def test_requested_non_aligned_dimensions_are_not_rounded(self):
        for dim in (64, 90, 102):
            with self.subTest(dim=dim):
                self.write_source(old_dim=128)
                self.output = self.root / f"plan-{dim}.json"
                with patch.object(subject.builder, "load_keep_maps", return_value=self.selection(dim)):
                    result = self.run_prepare(dim=dim)
                plan = json.loads(self.output.read_text(encoding="utf-8"))
                self.assertEqual(plan["target_head_k_dim"], dim)
                self.assertEqual(len(plan["keep_indices"]["0"]), 2 * dim)
                self.assertEqual(result["hf_overrides"]["text_config"]["linear_key_head_dim"], dim)

    def test_official_hash_drift_rejected_before_selection(self):
        self.official.write_text("changed", encoding="utf-8")
        with (
            patch.object(subject.builder, "load_keep_maps") as choose,
            self.assertRaisesRegex(ValueError, "official RRQR source hash drift"),
        ):
            self.run_prepare()
        choose.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_plan_is_untouched(self):
        self.output.write_text("existing evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing evidence")

    def test_failed_exclusive_write_cleans_partial_file(self):
        with (
            patch.object(subject.os, "fsync", side_effect=OSError("disk flush failed")),
            self.assertRaisesRegex(OSError, "disk flush failed"),
        ):
            subject.write_exclusive_json(self.output, {"example": True})
        self.assertFalse(self.output.exists())

    def test_output_cannot_modify_source_checkpoint(self):
        self.output = self.source / "plan.json"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_prepare()

    def test_dense_bf16_and_known_architecture_are_required(self):
        for key, value in (
            ("model_type", "qwen3_8_text"),
            ("dtype", "float16"),
            ("num_experts", 8),
            ("quantization_config", {"quant_method": "awq"}),
        ):
            with self.subTest(key=key):
                self.write_source()
                self.text_config[key] = value
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.run_prepare()
        self.write_source(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "BF16"):
            self.run_prepare()

    def test_missing_target_and_bias_are_rejected(self):
        key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        del self.index["weight_map"][key]
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_prepare()
        self.write_source()
        self.index["weight_map"][key.replace("weight", "bias")] = "model-00001-of-00001.safetensors"
        self.save_metadata()
        with self.assertRaisesRegex(ValueError, "bias"):
            self.run_prepare()

    def test_per_head_indices_are_required(self):
        maps, evidence = self.selection()
        maps[0] = torch.tensor([0, 4, 1, 5])
        with (
            patch.object(subject.builder, "load_keep_maps", return_value=(maps, evidence)),
            self.assertRaisesRegex(ValueError, "per-head"),
        ):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_source_metadata_change_during_selection_is_rejected(self):
        def changed_selection(*args, **kwargs):
            with (self.source / "config.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            return self.selection()

        with (
            patch.object(subject.builder, "load_keep_maps", side_effect=changed_selection),
            self.assertRaisesRegex(RuntimeError, "changed while"),
        ):
            self.run_prepare()
        self.assertFalse(self.output.exists())

    def test_actual_rrqr_selection_on_synthetic_capture(self):
        generator = torch.Generator().manual_seed(9)
        for layer in (0, 2):
            for rank in (0, 1):
                torch.save(
                    {
                        "prefix": f"model.layers.{layer}.linear_attn",
                        "tp_rank": rank,
                        "capture_index": 0,
                        "q": torch.randn(12, 4, generator=generator),
                        "k": torch.randn(12, 4, generator=generator),
                    },
                    self.capture / f"layer{layer}-rank{rank}.pt",
                )
        self.run_prepare()
        plan = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(plan["provenance"]["selection"]["capture_file_count"], 4)
        self.assertEqual(plan["provenance"]["selection"]["seed"], 42)
        self.assertEqual(len(plan["keep_indices"]["0"]), 4)
        self.assertEqual(len(plan["keep_indices"]["2"]), 4)


if __name__ == "__main__":
    unittest.main()

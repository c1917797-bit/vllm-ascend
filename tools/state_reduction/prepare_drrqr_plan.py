#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a DRRQR selection plan for load-time Ascend monkeypatching.

This tool never rewrites a checkpoint. Capture/model/calibration correspondence
must be established independently; hashing the supplied files alone does not
establish that the activations were collected from that model or dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import build_qwen36_drrqr_checkpoint as builder
from safetensors import safe_open

PLAN_SCHEMA = "ascend-drrqr-plan/v1"
SOURCE_PREFIXES = ("layers.", "model.layers.", "model.language_model.layers.", "language_model.model.layers.")
TARGET_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.linear_attn\.(in_proj_qkv|conv1d)\.(weight|bias)$")
SCOPE = (
    "Load-time structured Q/K selection for dense qwen3_5_text packed GDN; "
    "the model identity is taken from the source config. Selection reuses the "
    "existing load_keep_maps implementation, whose legacy evidence schema name "
    "is not a model identity. Source/config/index and supplied calibration hashes "
    "do not prove capture provenance, calibration/evaluation separation, paper "
    "reproduction, Ascend kernel compatibility, accuracy, or performance."
)


def sha256(path: Path) -> str:
    return builder.sha256(path)


def positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def validate_source(
    source: Path,
    config: dict[str, Any],
    index: dict[str, Any],
    new_dim: int,
    tp_size: int,
) -> tuple[dict[str, Any], list[int]]:
    """Check the source layout without allocating full checkpoint tensors."""
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict) or text_config.get("model_type") != "qwen3_5_text":
        raise ValueError("only dense qwen3_5_text configuration is supported")
    if config.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("unsupported top-level model_type")
    for candidate in (config, text_config):
        if candidate.get("quantization_config") not in (None, {}):
            raise ValueError("quantized checkpoints are not supported")
        if candidate.get("num_experts", 0):
            raise ValueError("MoE checkpoints are not supported")
    dtype = text_config.get("dtype", text_config.get("torch_dtype", config.get("dtype", config.get("torch_dtype"))))
    if dtype not in ("bfloat16", "torch.bfloat16"):
        raise ValueError("source configuration must specify bfloat16")

    old_dim = positive_int(text_config, "linear_key_head_dim")
    num_heads = positive_int(text_config, "linear_num_key_heads")
    num_value_heads = positive_int(text_config, "linear_num_value_heads")
    head_v_dim = positive_int(text_config, "linear_value_head_dim")
    conv_dim = positive_int(text_config, "linear_conv_kernel_dim")
    hidden_size = positive_int(text_config, "hidden_size")
    num_layers = positive_int(text_config, "num_hidden_layers")
    if type(new_dim) is not int or not 0 < new_dim < old_dim:
        raise ValueError("new head K dimension must be strictly between zero and the original dimension")
    if type(tp_size) is not int or tp_size <= 0 or num_heads % tp_size or num_value_heads % tp_size:
        raise ValueError("capture TP size must divide both key and value head counts")
    if num_value_heads % num_heads:
        raise ValueError("value head count must be divisible by key head count")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("layer_types must describe every source layer")
    if any(item not in ("linear_attention", "full_attention") for item in layer_types):
        raise ValueError("unsupported attention layer type")
    linear_layers = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    if not linear_layers or "full_attention" not in layer_types:
        raise ValueError("a hybrid model with both linear_attention and full_attention is required")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("a nonempty model.safetensors.index.json weight_map is required")

    targets: dict[tuple[int, str], tuple[str, str]] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("weight_map keys and shard names must be strings")
        match = TARGET_RE.search(name)
        if match is None or not name.startswith(SOURCE_PREFIXES):
            continue
        layer, component, kind = int(match[1]), match[2], match[3]
        if kind == "bias":
            raise ValueError(f"QKV/conv bias is not supported: {name}")
        if layer not in linear_layers:
            raise ValueError(f"linear-attention tensor outside configured linear layers: {name}")
        key = (layer, component)
        if key in targets:
            raise ValueError(f"duplicate source tensor for {key}")
        targets[key] = (name, shard)
    expected = {(layer, component) for layer in linear_layers for component in ("in_proj_qkv", "conv1d")}
    if set(targets) != expected:
        raise ValueError(f"packed source tensor coverage mismatch: missing={sorted(expected - set(targets))}")

    packed_width = 2 * num_heads * old_dim + num_value_heads * head_v_dim
    expected_shapes = {"in_proj_qkv": (packed_width, hidden_size), "conv1d": (packed_width, 1, conv_dim)}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for (_, component), (name, shard) in targets.items():
        by_shard.setdefault(shard, []).append((component, name))
    for shard, items in by_shard.items():
        shard_path = (source / shard).resolve()
        if not shard_path.is_relative_to(source.resolve()) or not shard_path.is_file():
            raise ValueError(f"source shard is missing or outside source directory: {shard}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for component, name in items:
                if name not in keys:
                    raise ValueError(f"indexed tensor missing from shard: {name}")
                tensor_slice = handle.get_slice(name)
                if tuple(tensor_slice.get_shape()) != expected_shapes[component]:
                    raise ValueError(f"source tensor shape mismatch: {name}")
                if tensor_slice.get_dtype() != "BF16":
                    raise ValueError(f"source target tensor must be BF16: {name}")
    return text_config, linear_layers


def write_exclusive_json(path: Path, value: object) -> None:
    """Publish once; refuse overwrites and remove our file if writing fails.

    Exclusive creation is not atomic publication: consumers should only use the
    path after this command completes successfully and prints its SHA-256.
    """
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def prepare_plan(
    src: Path,
    capture_dir: Path,
    calibration_jsonl: Path,
    official_rrqr: Path,
    new_head_k_dim: int,
    output: Path,
    *,
    tp_size: int = 4,
    captures_per_rank: int = 16,
) -> dict[str, Any]:
    source = Path(src).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"plan already exists: {output}")
    if output.is_relative_to(source):
        raise ValueError("plan output must be outside the immutable source checkpoint directory")
    if sha256(Path(official_rrqr)) != builder.OFFICIAL_RRQR_SHA256:
        raise ValueError("official RRQR source hash drift")
    if type(captures_per_rank) is not int or captures_per_rank <= 0:
        raise ValueError("captures_per_rank must be positive")
    calibration_path = Path(calibration_jsonl)
    if not calibration_path.is_file() or not calibration_path.stat().st_size:
        raise ValueError("a nonempty calibration JSONL file is required")
    if not Path(capture_dir).is_dir():
        raise ValueError("capture directory does not exist")

    config_path, index_path = source / "config.json", source / "model.safetensors.index.json"
    config_bytes, index_bytes = config_path.read_bytes(), index_path.read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)
    if not isinstance(config, dict) or not isinstance(index, dict):
        raise ValueError("source config and index must be JSON objects")
    text_config, linear_layers = validate_source(source, config, index, new_head_k_dim, tp_size)
    old_dim = text_config["linear_key_head_dim"]
    num_heads = text_config["linear_num_key_heads"]
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    index_sha = hashlib.sha256(index_bytes).hexdigest()
    calibration_sha = sha256(calibration_path)
    keep_maps, selection = builder.load_keep_maps(
        Path(capture_dir),
        expected_layer_ids=linear_layers,
        tp_size=tp_size,
        num_heads=num_heads,
        old_dim=old_dim,
        new_dim=new_head_k_dim,
        captures_per_rank=captures_per_rank,
    )
    if set(keep_maps) != set(linear_layers):
        raise ValueError("selection layer coverage mismatch")
    keep_indices: dict[str, list[int]] = {}
    for layer, tensor in keep_maps.items():
        values = tensor.tolist()
        if len(values) != num_heads * new_head_k_dim or any(type(item) is not int for item in values):
            raise ValueError(f"invalid selection length or type for layer {layer}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate selected coordinates for layer {layer}")
        for head in range(num_heads):
            selected = values[head * new_head_k_dim : (head + 1) * new_head_k_dim]
            if selected != sorted(selected) or any(
                not head * old_dim <= item < (head + 1) * old_dim for item in selected
            ):
                raise ValueError(f"selection violates per-head coordinates for layer {layer}, head {head}")
        keep_indices[str(layer)] = values
    if (
        sha256(config_path) != config_sha
        or sha256(index_path) != index_sha
        or sha256(calibration_path) != calibration_sha
    ):
        raise RuntimeError("source metadata or calibration changed while preparing the plan")

    plan = {
        "schema": PLAN_SCHEMA,
        "source_config_sha256": config_sha,
        "source_index_sha256": index_sha,
        "old_head_k_dim": old_dim,
        "target_head_k_dim": new_head_k_dim,
        "num_key_heads": num_heads,
        "num_value_heads": text_config["linear_num_value_heads"],
        "head_v_dim": text_config["linear_value_head_dim"],
        "conv_kernel_dim": text_config["linear_conv_kernel_dim"],
        "hidden_size": text_config["hidden_size"],
        "layer_types": text_config["layer_types"],
        "model_type": text_config["model_type"],
        "keep_indices": keep_indices,
        "provenance": {
            "method": "DRRQR",
            "official_commit": builder.OFFICIAL_COMMIT,
            "official_rrqr_sha256": builder.OFFICIAL_RRQR_SHA256,
            "calibration_sha256": calibration_sha,
            "selection": selection,
            "scope": SCOPE,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_exclusive_json(output, plan)
    plan_sha = sha256(output)
    overrides = {
        "linear_key_head_dim": new_head_k_dim,
        "ascend_drrqr": {"plan_path": str(output), "plan_sha256": plan_sha},
    }
    return {
        "plan_path": str(output),
        "plan_sha256": plan_sha,
        "hf_overrides": {"text_config": overrides} if "text_config" in config else overrides,
        "scope": SCOPE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("src", "capture-dir", "calibration-jsonl", "official-rrqr", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--new-head-k-dim", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=4, help="TP size used for the existing activation capture")
    parser.add_argument("--captures-per-rank", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(prepare_plan(**vars(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

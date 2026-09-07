#!/usr/bin/env python3
"""Build a Qwen3.6 packed-GDN checkpoint using the paper's Strong RRQR rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
OFFICIAL_COMMIT = "919d8667d951c385e08510bc1267c2e7049a4f56"
OFFICIAL_RRQR_SHA256 = "fa4bacf516011ba1c88f2e0e92957bbe4513cc1bb6634912fdb0d06ba7821a62"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def strong_rrqr_indices(
    activation: torch.Tensor, n_keep: int, f_param: float = 1.5, max_swaps: int = 20
) -> np.ndarray:
    """Faithful CPU adaptation of official rrqr.py strong_rrqr_indices."""
    matrix = activation.float().numpy()
    if matrix.shape[0] > 5000:
        chosen = np.random.choice(matrix.shape[0], 5000, replace=False)
        matrix = matrix[chosen]
    _, r_matrix, permutation = scipy.linalg.qr(
        matrix, pivoting=True, mode="economic"
    )
    permutation = permutation.copy()
    for _ in range(max_swaps):
        a_block = r_matrix[:n_keep, :n_keep]
        b_block = r_matrix[:n_keep, n_keep:]
        if np.abs(np.diag(a_block)).min() < 1e-9:
            break
        w_matrix = scipy.linalg.solve_triangular(a_block, b_block, lower=False)
        a_inverse = scipy.linalg.solve_triangular(
            a_block, np.eye(n_keep), lower=False
        )
        omega = 1.0 / (np.linalg.norm(a_inverse, axis=1) + 1e-12)
        if r_matrix.shape[0] > n_keep and r_matrix.shape[1] > n_keep:
            gamma = np.linalg.norm(r_matrix[n_keep:, n_keep:], axis=0)
        else:
            gamma = np.zeros(r_matrix.shape[1] - n_keep)
        rho = np.sqrt(
            np.abs(w_matrix) ** 2 + np.outer(1.0 / omega, gamma) ** 2
        )
        if np.max(rho) <= f_param:
            break
        left, right = np.unravel_index(np.argmax(rho), rho.shape)
        right += n_keep
        permutation[left], permutation[right] = (
            permutation[right],
            permutation[left],
        )
        _, r_matrix = scipy.linalg.qr(
            matrix[:, permutation], mode="economic"
        )
    return permutation[:n_keep]


def layer_id(name: str) -> int:
    match = LAYER_RE.search(name)
    if not match:
        raise ValueError(f"cannot parse layer from {name}")
    return int(match.group(1))


def load_keep_maps(
    capture_dir: Path,
    *,
    expected_layer_ids: list[int],
    tp_size: int,
    num_heads: int,
    old_dim: int,
    new_dim: int,
    captures_per_rank: int,
) -> tuple[dict[int, torch.Tensor], dict]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    capture_files = sorted(capture_dir.glob("*.pt"))
    for path in capture_files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        lid = layer_id(payload["prefix"])
        rank = int(payload["tp_rank"])
        grouped.setdefault((lid, rank), []).append(payload)
    expected_keys = {
        (layer, rank) for layer in expected_layer_ids for rank in range(tp_size)
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise RuntimeError(f"capture coverage mismatch missing={missing} extra={extra}")
    local_heads = num_heads // tp_size
    np.random.seed(42)
    keep_maps: dict[int, torch.Tensor] = {}
    evidence = {
        "schema": "qwen36-drrqr-selection/v1",
        "method": "Strong RRQR (Gu-Eisenstat) on concatenated post-conv Q/K activations",
        "official_commit": OFFICIAL_COMMIT,
        "official_rrqr_sha256": OFFICIAL_RRQR_SHA256,
        "f_param": 1.5,
        "max_swaps": 20,
        "subsample_threshold": 5000,
        "seed": 42,
        "capture_file_count": len(capture_files),
        "layers": {},
    }
    for lid in expected_layer_ids:
        global_keep = []
        layer_record = {"ranks": {}}
        for rank in range(tp_size):
            payloads = sorted(
                grouped[(lid, rank)], key=lambda item: int(item["capture_index"])
            )
            indexes = [int(item["capture_index"]) for item in payloads]
            if indexes != list(range(captures_per_rank)):
                raise RuntimeError(
                    f"layer {lid} rank {rank} capture indexes {indexes}"
                )
            q = torch.cat([item["q"] for item in payloads], dim=0)
            k = torch.cat([item["k"] for item in payloads], dim=0)
            if q.shape != k.shape or q.shape[1] != local_heads * old_dim:
                raise RuntimeError(
                    f"layer {lid} rank {rank} bad shape q={tuple(q.shape)} k={tuple(k.shape)}"
                )
            combined = torch.cat((q, k), dim=0).view(
                -1, local_heads, old_dim
            )
            rank_keep = []
            for local_head in range(local_heads):
                selected = strong_rrqr_indices(
                    combined[:, local_head, :], new_dim
                )
                selected = np.sort(selected)
                global_head = rank * local_heads + local_head
                values = (selected + global_head * old_dim).tolist()
                global_keep.extend(values)
                rank_keep.append(values)
            layer_record["ranks"][str(rank)] = {
                "captures": len(payloads),
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
                "keep_indices_by_local_head": rank_keep,
            }
        if len(global_keep) != num_heads * new_dim or len(set(global_keep)) != len(
            global_keep
        ):
            raise RuntimeError(f"invalid keep map for layer {lid}")
        keep_maps[lid] = torch.tensor(global_keep, dtype=torch.long)
        layer_record["global_keep_indices"] = global_keep
        evidence["layers"][str(lid)] = layer_record
    return keep_maps, evidence


def prune_packed(
    tensor: torch.Tensor, qk_width: int, keep: torch.Tensor
) -> torch.Tensor:
    if tensor.shape[0] <= 2 * qk_width:
        raise ValueError(f"unexpected packed tensor shape {tuple(tensor.shape)}")
    return torch.cat(
        (
            tensor[:qk_width].index_select(0, keep),
            tensor[qk_width : 2 * qk_width].index_select(0, keep),
            tensor[2 * qk_width :],
        ),
        dim=0,
    ).contiguous()


def hardlink_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=os.link)
        elif item.is_file():
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--official-rrqr", required=True)
    parser.add_argument("--new-head-k-dim", type=int, default=64)
    parser.add_argument("--expected-linear-layers", type=int, default=48)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--captures-per-rank", type=int, default=16)
    args = parser.parse_args()

    source, destination = Path(args.src), Path(args.dst)
    official = Path(args.official_rrqr)
    if sha256(official) != OFFICIAL_RRQR_SHA256:
        raise RuntimeError("official RRQR source hash drift")
    config = json.loads((source / "config.json").read_text())
    text_config = config["text_config"]
    old_dim = int(text_config["linear_key_head_dim"])
    new_dim = int(args.new_head_k_dim)
    num_heads = int(text_config["linear_num_key_heads"])
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        raise RuntimeError("text_config.layer_types is required for hybrid-layer coverage")
    linear_layer_ids = [
        layer for layer, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]
    if len(linear_layer_ids) != args.expected_linear_layers:
        raise RuntimeError(
            "linear layer count mismatch: "
            f"config={len(linear_layer_ids)} expected={args.expected_linear_layers}"
        )
    if num_heads % args.tp_size or not 0 < new_dim < old_dim:
        raise ValueError((num_heads, args.tp_size, old_dim, new_dim))
    qk_width = num_heads * old_dim
    if qk_width != 2048:
        raise RuntimeError(f"unexpected Q/K width {qk_width}")
    keep_maps, selection = load_keep_maps(
        Path(args.capture_dir),
        expected_layer_ids=linear_layer_ids,
        tp_size=args.tp_size,
        num_heads=num_heads,
        old_dim=old_dim,
        new_dim=new_dim,
        captures_per_rank=args.captures_per_rank,
    )
    selection["calibration_jsonl"] = str(Path(args.calibration_jsonl))
    selection["calibration_sha256"] = sha256(Path(args.calibration_jsonl))
    selection_path = Path(args.capture_dir).parent / "drrqr-selection-evidence.json"
    atomic_json(selection_path, selection)

    hardlink_copy(source, destination)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    manifest = {
        "schema": "paper-to-ascend-drrqr-checkpoint/v1",
        "method": "DRRQR",
        "model_id": "Qwen3.6-27B",
        "source_model_path": str(source),
        "source_config_sha256": sha256(source / "config.json"),
        "source_index_sha256": sha256(index_path),
        "official_repository": "https://github.com/camail-official/LinearAttentionPruning",
        "official_commit": OFFICIAL_COMMIT,
        "official_rrqr_sha256": OFFICIAL_RRQR_SHA256,
        "selection_evidence_path": str(selection_path),
        "selection_evidence_sha256": sha256(selection_path),
        "calibration_dataset_path": str(Path(args.calibration_jsonl)),
        "calibration_dataset_sha256": sha256(Path(args.calibration_jsonl)),
        "calibration_scope": "Ascend adaptation: 16 source-ordered LongBench-v2 rows x 2048 tokens; not FineWeb-Edu",
        "old_head_k_dim": old_dim,
        "target_head_k_dim": new_dim,
        "changed_shards": [],
        "changed_tensors": [],
        "changed_layers": len(linear_layer_ids),
        "changed_layer_ids": linear_layer_ids,
        "converter_version": "qwen36-packed-gdn-drrqr/v1",
    }
    for shard in shards:
        names = [name for name, mapped in weight_map.items() if mapped == shard]
        targets = [
            name
            for name in names
            if name.endswith(".linear_attn.in_proj_qkv.weight")
            or name.endswith(".linear_attn.conv1d.weight")
        ]
        if not targets:
            continue
        tensors = {}
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if name in targets:
                    lid = layer_id(name)
                    before = list(tensor.shape)
                    tensor = prune_packed(tensor, qk_width, keep_maps[lid])
                    manifest["changed_tensors"].append(
                        {
                            "name": name,
                            "shard": shard,
                            "before": before,
                            "after": list(tensor.shape),
                            "layer": lid,
                        }
                    )
                tensors[name] = tensor
        target_shard = destination / shard
        tmp = target_shard.with_name(target_shard.name + ".tmp")
        save_file(tensors, tmp, metadata=metadata)
        os.replace(tmp, target_shard)
        manifest["changed_shards"].append(shard)
        print(f"rewrote {shard}: targets={len(targets)}", flush=True)
    changed_layers = {item["layer"] for item in manifest["changed_tensors"]}
    if changed_layers != set(linear_layer_ids):
        raise RuntimeError(f"changed layer coverage mismatch: {sorted(changed_layers)}")
    if len(manifest["changed_tensors"]) != len(linear_layer_ids) * 2:
        raise RuntimeError(
            f"expected {len(linear_layer_ids) * 2} changed tensors, got "
            f"{len(manifest['changed_tensors'])}"
        )
    output_config = json.loads((source / "config.json").read_text())
    output_config["text_config"]["linear_key_head_dim"] = new_dim
    atomic_json(destination / "config.json", output_config)
    output_index = json.loads(index_path.read_text())
    if "metadata" in output_index and "total_size" in output_index["metadata"]:
        output_index["metadata"]["total_size"] = sum(
            (destination / shard).stat().st_size for shard in shards
        )
    atomic_json(destination / "model.safetensors.index.json", output_index)
    manifest["changed_tensor_count"] = len(manifest["changed_tensors"])
    manifest["checkpoint_config_sha256"] = sha256(destination / "config.json")
    manifest["checkpoint_index_sha256"] = sha256(
        destination / "model.safetensors.index.json"
    )
    from datetime import datetime, timezone

    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(destination / "drrqr_manifest.json", manifest)
    print(
        json.dumps(
            {
                "destination": str(destination),
                "changed_layers": len(changed_layers),
                "changed_tensors": manifest["changed_tensor_count"],
                "manifest_sha256": sha256(destination / "drrqr_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

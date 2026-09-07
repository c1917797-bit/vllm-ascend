#!/usr/bin/env python3
"""Offline contract test for the experiment-only RRQR Q/K capture hook.

The host does not need torch/torch_npu: this extracts the hook body from the
authoritative source file and executes it with a minimal tensor/save double.
It proves the default-disabled and per-layer/rank bounded-write properties;
real Ascend tensor behavior remains a separate calibration-run gate.
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.device = "cpu"

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def __getitem__(self, key: tuple[slice, slice]) -> "FakeTensor":
        rows, columns = key
        if rows != slice(None):
            raise AssertionError("unexpected row slice")
        return FakeTensor([row[columns] for row in self.rows])

    def detach(self) -> "FakeTensor":
        return self

    def to(self, *, device: str) -> "FakeTensor":
        if device != "cpu":
            raise AssertionError(f"unexpected target device: {device}")
        return self


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def save(payload: dict[str, Any], path: str) -> None:
        Path(path).write_bytes(pickle.dumps(payload))


def load_hook(source_path: Path) -> Any:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected: list[ast.stmt] = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if set(names) & {"_RRQR_CAPTURE_COUNTS", "_RRQR_CAPTURE_LOCK"}:
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_maybe_capture_rrqr_qk":
            node.decorator_list = []
            selected.append(node)
    if len(selected) != 3:
        raise AssertionError("capture globals/function not found exactly once")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {
        "os": os,
        "threading": threading,
        "time": time,
        "torch": FakeTorch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_maybe_capture_rrqr_qk"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn.py",
    )
    args = parser.parse_args()
    capture = load_hook(args.source)
    tensor = FakeTensor([list(range(8)) for _ in range(10)])
    managed = (
        "VLLM_ASCEND_GDN_RRQR_CAPTURE_ENABLE",
        "VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR",
        "VLLM_ASCEND_GDN_RRQR_MAX_TOKENS",
        "VLLM_ASCEND_GDN_RRQR_MAX_CAPTURES_PER_LAYER",
    )
    saved = {key: os.environ.get(key) for key in managed}
    try:
        for key in managed:
            os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            os.environ["VLLM_ASCEND_GDN_RRQR_CAPTURE_DIR"] = directory
            capture(
                tensor, prefix="layer.0", tp_rank=0, local_key_dim=2, head_k_dim=2
            )
            assert not list(Path(directory).glob("*.pt")), "directory alone armed capture"

            os.environ["VLLM_ASCEND_GDN_RRQR_CAPTURE_ENABLE"] = "1"
            capture(
                tensor,
                prefix="layer.0",
                tp_rank=0,
                local_key_dim=2,
                head_k_dim=2,
            )
            assert not list(Path(directory).glob("*.pt")), "capture armed without sentinel"
            (Path(directory) / ".armed").touch()
            for _ in range(2):
                capture(
                    tensor,
                    prefix="layer.0",
                    tp_rank=0,
                    local_key_dim=2,
                    head_k_dim=2,
                )
            files = list(Path(directory).glob("*.pt"))
            assert len(files) == 1, f"expected one bounded capture, got {files}"
            payload = pickle.loads(files[0].read_bytes())
            assert payload["q"].shape == (10, 2)
            assert payload["k"].shape == (10, 2)
            assert payload["capture_index"] == 0
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("RRQR capture hook offline contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

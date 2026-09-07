import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools" / "state_reduction"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCalibrationHelpers(unittest.TestCase):
    def test_atomic_jsonl_is_compact_and_deterministic(self):
        calibration = load_module(
            "state_reduction_calibration", "state_reduction_calibration.py"
        )
        rows = [{"index": 0, "text": "昇腾"}, {"index": 1, "value": 3}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "calibration.jsonl"
            calibration.atomic_jsonl(path, rows)
            expected = (
                '{"index":0,"text":"昇腾"}\n'
                '{"index":1,"value":3}\n'
            ).encode()
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(
                calibration.sha256_bytes(expected),
                "88f65a3804953e1992b323c6468810714f12c6397a44314b7ab9685ad2b745a9",
            )
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


class TestCaptureInstaller(unittest.TestCase):
    def test_replace_once_replaces_only_anchor(self):
        installer = load_module(
            "install_gdn_rrqr_capture_v023",
            "install_gdn_rrqr_capture_v023.py",
        )
        self.assertEqual(
            installer.replace_once("before ANCHOR after", "ANCHOR", "new"),
            "before new after",
        )

    def test_replace_once_rejects_missing_or_ambiguous_anchor(self):
        installer = load_module(
            "install_gdn_rrqr_capture_v023_errors",
            "install_gdn_rrqr_capture_v023.py",
        )
        with self.assertRaises(RuntimeError):
            installer.replace_once("no match", "ANCHOR", "new")
        with self.assertRaises(RuntimeError):
            installer.replace_once("ANCHOR ANCHOR", "ANCHOR", "new")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.io.run_bundle import (
    build_result_filename,
    mark_run_completed,
    mark_run_failed,
    prepare_run_bundle,
    sha256_file,
)


class RunBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "model_type": "model_empirical",
            "seed": 17,
            "data": {"angles": [45, 90, 135], "direction": "v"},
            "bias": {
                "add_bias_E1": False,
                "add_bias_alpha": False,
            },
        }

    def test_prepare_run_bundle_writes_running_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.io.run_bundle.result_output_root", return_value=Path(directory)):
                bundle = prepare_run_bundle(self.config, "experimental", ["python", "main.py"])

            manifest = json.loads(bundle.manifest_path.read_text())
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["seed"], 17)
            self.assertEqual(manifest["command"], ["python", "main.py"])
            self.assertEqual(manifest["result"]["path"], bundle.result_path.name)
            self.assertIn("config", manifest)

    def test_mark_run_completed_records_checksum_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.io.run_bundle.result_output_root", return_value=Path(directory)):
                bundle = prepare_run_bundle(self.config, "final", ["python", "main.py", "--final"])

            bundle.result_path.write_bytes(b"posterior")
            manifest = mark_run_completed(bundle)

            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["result"]["path"], bundle.result_path.name)
            self.assertEqual(manifest["result"]["size_bytes"], len(b"posterior"))
            self.assertEqual(manifest["result"]["sha256"], sha256_file(bundle.result_path))

    def test_mark_run_failed_preserves_bundle_and_sets_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.io.run_bundle.result_output_root", return_value=Path(directory)):
                bundle = prepare_run_bundle(self.config, "default", ["python", "main.py"])

            manifest = mark_run_failed(bundle, "sampler exploded")

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "sampler exploded")

    def test_build_result_filename_includes_direction_for_non_hv_models(self) -> None:
        filename = build_result_filename(self.config, "20260615T120000000000Z")
        self.assertTrue(filename.startswith("no_bias_normal"))
        self.assertTrue(filename.endswith("_MAF_linear.nc"))


if __name__ == "__main__":
    unittest.main()

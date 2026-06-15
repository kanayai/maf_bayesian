from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpyro.distributions as dist

from src.io.result_selection import existing_analysis_source, existing_netcdf_path
from src.io.run_bundle import sha256_file


class ExistingNetcdfPathTests(unittest.TestCase):
    def test_returns_resolved_explicit_netcdf_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "posterior.nc"
            result.touch()

            self.assertEqual(existing_netcdf_path(str(result)), result.resolve())

    def test_rejects_missing_result(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "does not exist"):
            existing_netcdf_path("missing.nc")

    def test_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(argparse.ArgumentTypeError, "not a file"):
                existing_netcdf_path(directory)

    def test_rejects_non_netcdf_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "posterior.csv"
            result.touch()

            with self.assertRaisesRegex(argparse.ArgumentTypeError, ".nc extension"):
                existing_netcdf_path(str(result))


class ExistingAnalysisSourceTests(unittest.TestCase):
    def _write_bundle(self, bundle_dir: Path, *, run_id: str = "run_001", status: str = "completed") -> Path:
        result_path = bundle_dir / "posterior.nc"
        result_path.write_bytes(b"posterior")
        manifest = {
            "run_id": run_id,
            "status": status,
            "result": {
                "path": "posterior.nc",
                "sha256": sha256_file(result_path),
                "size_bytes": len(b"posterior"),
            },
            "config": {
                "model_type": "model_empirical",
                "data": {"angles": [45, 90, 135]},
                "priors": {
                    "hyper": {
                        "sigma_measure_base": {"target_dist": "dist.Exponential(100.0)"}
                    }
                },
            },
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
        return result_path

    def test_accepts_bundle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory)
            result_path = self._write_bundle(bundle_dir)

            source = existing_analysis_source(str(bundle_dir))

            self.assertEqual(source.bundle_dir, bundle_dir.resolve())
            self.assertEqual(source.result_path, result_path.resolve())
            self.assertEqual(source.run_id, "run_001")
            self.assertIsInstance(
                source.config["priors"]["hyper"]["sigma_measure_base"]["target_dist"],
                dist.Exponential,
            )

    def test_accepts_bundled_netcdf_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = self._write_bundle(Path(directory))

            source = existing_analysis_source(str(result_path))

            self.assertEqual(source.result_path, result_path.resolve())

    def test_rejects_unbundled_netcdf_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "posterior.nc"
            result_path.touch()

            with self.assertRaisesRegex(argparse.ArgumentTypeError, "manifest.json"):
                existing_analysis_source(str(result_path))

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory)
            result_path = self._write_bundle(bundle_dir)
            result_path.write_bytes(b"tampered")

            with self.assertRaisesRegex(argparse.ArgumentTypeError, "checksum mismatch"):
                existing_analysis_source(str(bundle_dir))

    def test_rejects_incomplete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory)
            self._write_bundle(bundle_dir, status="failed")

            with self.assertRaisesRegex(argparse.ArgumentTypeError, "not completed"):
                existing_analysis_source(str(bundle_dir))

    def test_resolves_run_id_under_results_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir = root / "results" / "final" / "run_abc"
            bundle_dir.mkdir(parents=True)
            self._write_bundle(bundle_dir, run_id="run_abc")

            with patch(
                "src.io.result_selection.Path",
                side_effect=lambda *parts: root / Path(*parts),
            ):
                source = existing_analysis_source("run_abc")

            self.assertEqual(source.run_id, "run_abc")
            self.assertEqual(source.bundle_dir, bundle_dir.resolve())

    def test_rejects_ambiguous_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_a = root / "results" / "tmp" / "run_dup"
            bundle_b = root / "results" / "final" / "run_dup"
            bundle_a.mkdir(parents=True)
            bundle_b.mkdir(parents=True)
            self._write_bundle(bundle_a, run_id="run_dup")
            self._write_bundle(bundle_b, run_id="run_dup")

            with patch(
                "src.io.result_selection.Path",
                side_effect=lambda *parts: root / Path(*parts),
            ):
                with self.assertRaisesRegex(argparse.ArgumentTypeError, "ambiguous"):
                    existing_analysis_source("run_dup")


if __name__ == "__main__":
    unittest.main()

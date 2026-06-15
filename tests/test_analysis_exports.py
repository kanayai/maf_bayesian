from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.io.analysis_exports import (
    compute_diagnostics_summary,
    ensure_exports_dir,
    flatten_prediction_collection,
    summarize_samples,
    write_acceptance_summary,
    write_analysis_manifest,
    write_prediction_exports,
    write_posterior_summary,
    write_residual_exports,
)


class AnalysisExportsTests(unittest.TestCase):
    def test_summarize_samples_includes_quantiles(self) -> None:
        df = summarize_samples({"theta": np.array([1.0, 2.0, 3.0, 4.0])})
        self.assertEqual(list(df["parameter"]), ["theta"])
        self.assertEqual(int(df.loc[0, "n_samples"]), 4)
        self.assertAlmostEqual(float(df.loc[0, "mean"]), 2.5)
        self.assertAlmostEqual(float(df.loc[0, "q05"]), 1.15)
        self.assertAlmostEqual(float(df.loc[0, "q95"]), 3.85)

    def test_write_posterior_summary_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exports_dir = ensure_exports_dir(Path(directory))
            paths = write_posterior_summary({"theta": np.array([1.0, 2.0, 3.0])}, exports_dir)
            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())

    def test_compute_diagnostics_summary_reports_sample_stats(self) -> None:
        idata = az.from_dict(
            posterior={"theta": np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]])},
            sample_stats={
                "diverging": np.array([[0, 1, 0, 0], [0, 0, 0, 0]]),
                "acceptance_rate": np.array([[0.7, 0.8, 0.9, 1.0], [0.75, 0.85, 0.95, 1.0]]),
                "lp": np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]]),
            },
        )
        diagnostics = compute_diagnostics_summary(idata)
        self.assertEqual(diagnostics["total_divergences"], 1)
        self.assertAlmostEqual(diagnostics["mean_acceptance_rate"], 0.86875)
        self.assertEqual(diagnostics["n_chains"], 2)
        self.assertEqual(diagnostics["n_draws"], 4)

    def test_flatten_prediction_collection_produces_prediction_and_observation_rows(self) -> None:
        predictions_collection = {
            45: {
                "v": {
                    "samples_load": np.array([0.0, 1.0]),
                    "mean_prior": np.array([0.1, 0.2]),
                    "mean_post": np.array([0.3, 0.4]),
                    "pct_prior_f": np.array([[0.0, 0.1], [0.2, 0.3]]),
                    "pct_prior_y": np.array([[0.0, 0.05], [0.25, 0.35]]),
                    "pct_post_f": np.array([[0.2, 0.3], [0.4, 0.5]]),
                    "pct_post_y": np.array([[0.15, 0.25], [0.45, 0.55]]),
                    "training_info": "trained",
                    "input_xy_exp": [np.array([[0.0, 0.1], [1.0, 0.1]])],
                    "data_exp": [np.array([[1.0, 1.1], [2.0, 2.1]])],
                }
            }
        }
        prediction_df, observation_df = flatten_prediction_collection(predictions_collection)
        self.assertEqual(len(prediction_df), 2)
        self.assertEqual(len(observation_df), 4)
        self.assertIn("posterior_observation_upper", prediction_df.columns)
        self.assertIn("observed_extension", observation_df.columns)

    def test_write_residual_exports_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exports_dir = ensure_exports_dir(Path(directory))
            residual_paths = write_residual_exports(
                residual_rows=[{"Angle": 45, "Direction": "Normal", "Load": 1.0, "Residual": 0.1, "StdResidual": 0.2}],
                bands_data=[{"Angle": 45, "Direction": "Normal", "Load": np.array([0.0, 1.0]), "SigmaTotal": np.array([0.5, 0.6])}],
                exports_dir=exports_dir,
            )
            self.assertEqual(len(residual_paths), 2)
            manifest_path = write_analysis_manifest(
                exports_dir,
                run_id="run_001",
                manifest_path=Path("/tmp/run_001/manifest.json"),
                result_path=Path("/tmp/run_001/posterior.nc"),
                figures_dir=Path("/tmp/figures/analysis_001"),
                output_mode="final",
                exported_files=residual_paths,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["run_id"], "run_001")
            self.assertEqual(len(manifest["exported_files"]), 2)

    def test_write_prediction_exports_writes_both_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exports_dir = ensure_exports_dir(Path(directory))
            predictions_collection = {
                90: {
                    "h": {
                        "samples_load": np.array([0.0]),
                        "mean_prior": np.array([0.1]),
                        "mean_post": np.array([0.2]),
                        "pct_prior_f": np.array([[0.0], [0.2]]),
                        "pct_prior_y": np.array([[0.0], [0.3]]),
                        "pct_post_f": np.array([[0.1], [0.3]]),
                        "pct_post_y": np.array([[0.05], [0.35]]),
                        "training_info": None,
                        "input_xy_exp": [np.array([[0.0, 1.0]])],
                        "data_exp": [np.array([[1.0, 1.1, 1.2]])],
                    }
                }
            }
            paths = write_prediction_exports(predictions_collection, exports_dir)
            self.assertEqual(len(paths), 2)
            self.assertTrue(paths[0].exists())
            self.assertTrue(paths[1].exists())

    def test_write_acceptance_summary_writes_gate_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exports_dir = ensure_exports_dir(Path(directory))
            diagnostics_summary = {
                "parameters": [{"parameter": "theta", "ess_bulk": 150.0, "ess_tail": 120.0, "r_hat": 1.01}],
                "total_divergences": 0,
                "mean_acceptance_rate": 0.9,
            }
            prediction_df = pd.DataFrame(
                [
                    {
                        "angle_deg": 45,
                        "direction": "v",
                        "load": 0.0,
                        "posterior_observation_lower": 0.0,
                        "posterior_observation_upper": 1.0,
                        "posterior_function_lower": 0.1,
                        "posterior_function_upper": 0.9,
                    }
                ]
            )
            observation_df = pd.DataFrame(
                [
                    {
                        "angle_deg": 45,
                        "direction": "v",
                        "load": 0.0,
                        "replicate_index": 0,
                        "observed_extension": 0.5,
                    }
                ]
            )
            summary_path = write_acceptance_summary(
                run_id="run_001",
                diagnostics_summary=diagnostics_summary,
                prediction_df=prediction_df,
                observation_df=observation_df,
                exports_dir=exports_dir,
            )
            summary = json.loads(summary_path.read_text())
            self.assertTrue(summary["all_gates_passed"])
            self.assertEqual(summary["run_id"], "run_001")


if __name__ == "__main__":
    unittest.main()

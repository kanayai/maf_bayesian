from __future__ import annotations

import unittest

import pandas as pd

from src.io.acceptance_rules import (
    build_acceptance_summary,
    validate_acceptance_summary_document,
)


class AcceptanceRulesTests(unittest.TestCase):
    def test_build_acceptance_summary_passes_when_all_gates_clear(self) -> None:
        diagnostics_summary = {
            "parameters": [
                {"parameter": "theta", "ess_bulk": 200.0, "ess_tail": 180.0, "r_hat": 1.01},
                {"parameter": "sigma", "ess_bulk": 150.0, "ess_tail": 140.0, "r_hat": 1.02},
            ],
            "total_divergences": 0,
            "mean_acceptance_rate": 0.9,
            "n_chains": 1,
            "n_draws": 100,
        }
        prediction_df = pd.DataFrame(
            [
                {
                    "angle_deg": 45,
                    "direction": "v",
                    "load": 1.0,
                    "posterior_observation_lower": 0.0,
                    "posterior_observation_upper": 2.0,
                    "posterior_function_lower": 0.2,
                    "posterior_function_upper": 1.8,
                }
            ]
        )
        observation_df = pd.DataFrame(
            [
                {
                    "angle_deg": 45,
                    "direction": "v",
                    "load": 1.0,
                    "replicate_index": 0,
                    "observed_extension": 1.0,
                }
            ]
        )

        summary = build_acceptance_summary(
            run_id="run_001",
            diagnostics_summary=diagnostics_summary,
            prediction_df=prediction_df,
            observation_df=observation_df,
        )

        self.assertTrue(summary["all_gates_passed"])
        self.assertAlmostEqual(summary["metrics"]["posterior_observation_coverage"], 1.0)
        validate_acceptance_summary_document(summary, expected_run_id="run_001")

    def test_validate_acceptance_summary_rejects_failed_gate(self) -> None:
        summary = {
            "gates_version": 1,
            "run_id": "run_002",
            "all_gates_passed": False,
            "gate_results": [{"name": "max_rhat", "passed": False}],
        }

        with self.assertRaisesRegex(ValueError, "does not pass all required gates"):
            validate_acceptance_summary_document(summary, expected_run_id="run_002")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


ACCEPTANCE_GATES_VERSION = 1
DEFAULT_ACCEPTANCE_GATES = {
    "max_total_divergences": 0,
    "max_rhat": 1.05,
    "min_ess_bulk": 100.0,
    "min_ess_tail": 100.0,
    "min_posterior_observation_coverage": 0.80,
}


def summarize_diagnostics_metrics(diagnostics_summary: dict[str, Any]) -> dict[str, Any]:
    parameters = diagnostics_summary.get("parameters", [])
    rhats = []
    ess_bulk = []
    ess_tail = []

    for parameter in parameters:
        r_hat = parameter.get("r_hat")
        if _is_finite_number(r_hat):
            rhats.append(float(r_hat))

        bulk = parameter.get("ess_bulk")
        if _is_finite_number(bulk):
            ess_bulk.append(float(bulk))

        tail = parameter.get("ess_tail")
        if _is_finite_number(tail):
            ess_tail.append(float(tail))

    return {
        "total_divergences": int(diagnostics_summary.get("total_divergences", 0)),
        "mean_acceptance_rate": _optional_float(diagnostics_summary.get("mean_acceptance_rate")),
        "max_rhat": max(rhats) if rhats else None,
        "min_ess_bulk": min(ess_bulk) if ess_bulk else None,
        "min_ess_tail": min(ess_tail) if ess_tail else None,
        "n_chains": diagnostics_summary.get("n_chains"),
        "n_draws": diagnostics_summary.get("n_draws"),
        "n_diagnostic_parameters": len(parameters),
    }


def summarize_prediction_coverage(
    prediction_df: pd.DataFrame,
    observation_df: pd.DataFrame,
) -> dict[str, Any]:
    if prediction_df.empty or observation_df.empty:
        return {
            "observation_count": 0,
            "posterior_observation_coverage": None,
            "posterior_function_coverage": None,
        }

    coverage_rows = []
    grouped_predictions = {
        key: frame.sort_values("load").reset_index(drop=True)
        for key, frame in prediction_df.groupby(["angle_deg", "direction"])
    }

    for observation in observation_df.itertuples(index=False):
        key = (int(observation.angle_deg), observation.direction)
        prediction_rows = grouped_predictions.get(key)
        if prediction_rows is None or prediction_rows.empty:
            raise ValueError(
                "Acceptance coverage could not find prediction rows for "
                f"angle={observation.angle_deg}, direction={observation.direction}"
            )

        nearest_idx = (prediction_rows["load"] - float(observation.load)).abs().idxmin()
        nearest = prediction_rows.loc[nearest_idx]
        coverage_rows.append(
            {
                "observed_extension": float(observation.observed_extension),
                "posterior_observation_lower": float(nearest["posterior_observation_lower"]),
                "posterior_observation_upper": float(nearest["posterior_observation_upper"]),
                "posterior_function_lower": float(nearest["posterior_function_lower"]),
                "posterior_function_upper": float(nearest["posterior_function_upper"]),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    observed = coverage_df["observed_extension"]
    posterior_obs_hits = (
        (observed >= coverage_df["posterior_observation_lower"])
        & (observed <= coverage_df["posterior_observation_upper"])
    )
    posterior_fun_hits = (
        (observed >= coverage_df["posterior_function_lower"])
        & (observed <= coverage_df["posterior_function_upper"])
    )

    return {
        "observation_count": int(len(coverage_df)),
        "posterior_observation_coverage": float(posterior_obs_hits.mean()),
        "posterior_function_coverage": float(posterior_fun_hits.mean()),
    }


def build_acceptance_summary(
    *,
    run_id: str,
    diagnostics_summary: dict[str, Any],
    prediction_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_gates = dict(DEFAULT_ACCEPTANCE_GATES if gates is None else gates)
    diagnostics_metrics = summarize_diagnostics_metrics(diagnostics_summary)
    coverage_metrics = summarize_prediction_coverage(prediction_df, observation_df)
    metrics = {**diagnostics_metrics, **coverage_metrics}

    gate_results = [
        _evaluate_gate(
            name="max_total_divergences",
            observed=metrics.get("total_divergences"),
            comparator="<=",
            threshold=active_gates["max_total_divergences"],
        ),
        _evaluate_gate(
            name="max_rhat",
            observed=metrics.get("max_rhat"),
            comparator="<=",
            threshold=active_gates["max_rhat"],
        ),
        _evaluate_gate(
            name="min_ess_bulk",
            observed=metrics.get("min_ess_bulk"),
            comparator=">=",
            threshold=active_gates["min_ess_bulk"],
        ),
        _evaluate_gate(
            name="min_ess_tail",
            observed=metrics.get("min_ess_tail"),
            comparator=">=",
            threshold=active_gates["min_ess_tail"],
        ),
        _evaluate_gate(
            name="min_posterior_observation_coverage",
            observed=metrics.get("posterior_observation_coverage"),
            comparator=">=",
            threshold=active_gates["min_posterior_observation_coverage"],
        ),
    ]

    return {
        "gates_version": ACCEPTANCE_GATES_VERSION,
        "run_id": run_id,
        "metrics": metrics,
        "gates": active_gates,
        "gate_results": gate_results,
        "all_gates_passed": all(result["passed"] for result in gate_results),
    }


def validate_acceptance_summary_document(
    summary: dict[str, Any],
    *,
    expected_run_id: str,
    gates_version: int = ACCEPTANCE_GATES_VERSION,
) -> dict[str, Any]:
    if summary.get("gates_version") != gates_version:
        raise ValueError(
            f"Acceptance summary gates_version must be {gates_version}: {summary.get('gates_version')}"
        )
    if summary.get("run_id") != expected_run_id:
        raise ValueError(
            "Acceptance summary run_id does not match registry source_run_id: "
            f"{summary.get('run_id')} != {expected_run_id}"
        )
    if summary.get("all_gates_passed") is not True:
        raise ValueError("Acceptance summary does not pass all required gates")

    gate_results = summary.get("gate_results")
    if not isinstance(gate_results, list) or not gate_results:
        raise ValueError("Acceptance summary must include non-empty gate_results")
    for result in gate_results:
        if result.get("passed") is not True:
            raise ValueError(
                "Acceptance summary contains a failed gate: "
                f"{result.get('name', 'unknown')}"
            )

    return summary


def load_and_validate_acceptance_summary(
    path: Path,
    *,
    expected_run_id: str,
    gates_version: int = ACCEPTANCE_GATES_VERSION,
) -> dict[str, Any]:
    import json

    summary = json.loads(path.read_text())
    return validate_acceptance_summary_document(
        summary,
        expected_run_id=expected_run_id,
        gates_version=gates_version,
    )


def _evaluate_gate(*, name: str, observed: Any, comparator: str, threshold: float) -> dict[str, Any]:
    if not _is_finite_number(observed):
        return {
            "name": name,
            "observed": observed,
            "comparator": comparator,
            "threshold": threshold,
            "passed": False,
        }

    observed_value = float(observed)
    if comparator == "<=":
        passed = observed_value <= float(threshold)
    elif comparator == ">=":
        passed = observed_value >= float(threshold)
    else:
        raise ValueError(f"Unsupported comparator: {comparator}")

    return {
        "name": name,
        "observed": observed_value,
        "comparator": comparator,
        "threshold": float(threshold),
        "passed": passed,
    }


def _is_finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    return float(value) if _is_finite_number(value) else None

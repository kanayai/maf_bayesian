from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import jax.numpy as jnp
import numpy as np
import pandas as pd


def ensure_exports_dir(figures_dir: Path) -> Path:
    exports_dir = figures_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir


def summarize_samples(samples_dict: dict[str, Any]) -> pd.DataFrame:
    records = []
    for key, val in samples_dict.items():
        arr = np.asarray(val)
        flat = arr.reshape(-1)
        records.append(
            {
                "parameter": key,
                "n_samples": int(flat.size),
                "mean": float(np.mean(flat)),
                "variance": float(np.var(flat)),
                "std": float(np.std(flat)),
                "median": float(np.median(flat)),
                "q05": float(np.quantile(flat, 0.05)),
                "q95": float(np.quantile(flat, 0.95)),
            }
        )
    return pd.DataFrame(records)


def write_posterior_summary(
    samples_dict: dict[str, Any],
    exports_dir: Path,
    filename_stem: str = "posterior_summary",
) -> list[Path]:
    df = summarize_samples(samples_dict).sort_values("parameter").reset_index(drop=True)
    csv_path = exports_dir / f"{filename_stem}.csv"
    json_path = exports_dir / f"{filename_stem}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2) + "\n")
    return [csv_path, json_path]


def compute_diagnostics_summary(idata: az.InferenceData) -> dict[str, Any]:
    summary = az.summary(idata, kind="diagnostics", round_to=None)
    diagnostics: dict[str, Any] = {
        "parameters": json.loads(summary.reset_index(names="parameter").to_json(orient="records"))
    }

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None:
        if "diverging" in sample_stats:
            divergences = np.asarray(sample_stats["diverging"].values)
            diagnostics["total_divergences"] = int(divergences.sum())
        if "acceptance_rate" in sample_stats:
            acceptance = np.asarray(sample_stats["acceptance_rate"].values)
            diagnostics["mean_acceptance_rate"] = float(np.mean(acceptance))
        if "lp" in sample_stats:
            lp = np.asarray(sample_stats["lp"].values)
            diagnostics["mean_log_posterior"] = float(np.mean(lp))

    posterior = getattr(idata, "posterior", None)
    if posterior is not None:
        diagnostics["n_chains"] = int(posterior.sizes.get("chain", 0))
        diagnostics["n_draws"] = int(posterior.sizes.get("draw", 0))

    return diagnostics


def write_diagnostics_summary(idata: az.InferenceData, exports_dir: Path) -> Path:
    diagnostics_path = exports_dir / "diagnostics_summary.json"
    diagnostics_path.write_text(json.dumps(compute_diagnostics_summary(idata), indent=2) + "\n")
    return diagnostics_path


def flatten_prediction_collection(
    predictions_collection: dict[int, dict[str, dict[str, Any]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []

    for angle, direction_map in predictions_collection.items():
        for direction, payload in direction_map.items():
            loads = np.asarray(payload["samples_load"])
            mean_prior = np.asarray(payload["mean_prior"])
            mean_post = np.asarray(payload["mean_post"])
            pct_prior_f = np.asarray(payload["pct_prior_f"])
            pct_prior_y = np.asarray(payload["pct_prior_y"])
            pct_post_f = np.asarray(payload["pct_post_f"])
            pct_post_y = np.asarray(payload["pct_post_y"])

            for idx, load in enumerate(loads):
                prediction_rows.append(
                    {
                        "angle_deg": int(angle),
                        "direction": direction,
                        "load": float(load),
                        "mean_prior": float(mean_prior[idx]),
                        "prior_function_lower": float(pct_prior_f[0, idx]),
                        "prior_function_upper": float(pct_prior_f[1, idx]),
                        "prior_observation_lower": float(pct_prior_y[0, idx]),
                        "prior_observation_upper": float(pct_prior_y[1, idx]),
                        "mean_posterior": float(mean_post[idx]),
                        "posterior_function_lower": float(pct_post_f[0, idx]),
                        "posterior_function_upper": float(pct_post_f[1, idx]),
                        "posterior_observation_lower": float(pct_post_y[0, idx]),
                        "posterior_observation_upper": float(pct_post_y[1, idx]),
                        "training_info": payload.get("training_info"),
                    }
                )

            for xy, raw_data in zip(payload["input_xy_exp"], payload["data_exp"]):
                xy_arr = np.asarray(xy)
                raw_arr = np.asarray(raw_data)
                if raw_arr.ndim == 1:
                    raw_arr = raw_arr[:, None]
                for row_idx in range(raw_arr.shape[0]):
                    for rep_idx in range(raw_arr.shape[1]):
                        observation_rows.append(
                            {
                                "angle_deg": int(angle),
                                "direction": direction,
                                "load": float(xy_arr[row_idx, 0]),
                                "replicate_index": int(rep_idx),
                                "observed_extension": float(raw_arr[row_idx, rep_idx]),
                            }
                        )

    return pd.DataFrame(prediction_rows), pd.DataFrame(observation_rows)


def write_prediction_exports(
    predictions_collection: dict[int, dict[str, dict[str, Any]]],
    exports_dir: Path,
) -> list[Path]:
    prediction_df, observation_df = flatten_prediction_collection(predictions_collection)
    prediction_path = exports_dir / "prediction_plot_data.csv"
    observation_path = exports_dir / "experimental_observations.csv"
    prediction_df.to_csv(prediction_path, index=False)
    observation_df.to_csv(observation_path, index=False)
    return [prediction_path, observation_path]


def write_residual_exports(
    residual_rows: list[dict[str, Any]],
    bands_data: list[dict[str, Any]],
    exports_dir: Path,
) -> list[Path]:
    residuals_path = exports_dir / "residual_observations.csv"
    bands_path = exports_dir / "residual_bands.csv"

    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(residuals_path, index=False)

    band_rows: list[dict[str, Any]] = []
    for band in bands_data:
        loads = np.asarray(band["Load"])
        sigma_total = np.asarray(band["SigmaTotal"])
        for idx, load in enumerate(loads):
            band_rows.append(
                {
                    "angle_deg": int(band["Angle"]),
                    "direction": band["Direction"],
                    "load": float(load),
                    "sigma_total": float(sigma_total[idx]),
                    "lower_95": float(-1.96 * sigma_total[idx]),
                    "upper_95": float(1.96 * sigma_total[idx]),
                }
            )
    pd.DataFrame(band_rows).to_csv(bands_path, index=False)
    return [residuals_path, bands_path]


def write_analysis_manifest(
    exports_dir: Path,
    *,
    run_id: str,
    manifest_path: Path,
    result_path: Path,
    figures_dir: Path,
    output_mode: str,
    exported_files: list[Path],
) -> Path:
    manifest = {
        "run_id": run_id,
        "source_manifest": str(manifest_path),
        "source_result": str(result_path),
        "analysis_output_dir": str(figures_dir),
        "output_mode": output_mode,
        "exported_files": [str(path) for path in exported_files],
    }
    manifest_path_out = exports_dir / "analysis_manifest.json"
    manifest_path_out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path_out

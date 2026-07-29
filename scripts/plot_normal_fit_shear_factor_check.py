import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.default_config import config
from src.io.data_loader import load_all_data


def fit_origin_slope(load, extension):
    """Least-squares fit for extension = beta * load through the origin."""
    if extension.ndim == 2:
        load = np.repeat(load, extension.shape[1])
        extension = extension.reshape(-1)
    denominator = np.sum(load**2)
    if denominator <= 0:
        return np.nan
    return np.sum(load * extension) / denominator


def collect_angle_data(input_xy_exp, extensions, angle):
    rows = []
    for exp_idx, xy in enumerate(input_xy_exp):
        exp_angle = int(round(np.rad2deg(float(xy[0, 1]))))
        if exp_angle != angle:
            continue
        load = np.asarray(xy[:, 0], dtype=float)
        extension = np.asarray(extensions[exp_idx], dtype=float)
        rows.append((exp_idx, load, extension))
    return rows


def main():
    data = load_all_data(config)
    input_xy_exp = data["input_xy_exp"]
    normal_data = data["data_exp_v"]
    shear_data = data["data_exp_h"]
    angles = [45, 90, 135]
    direct_shear_angle = 90

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        len(angles),
        figsize=(15, 7),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    rows = []

    for col_idx, angle in enumerate(angles):
        normal_matches = collect_angle_data(input_xy_exp, normal_data, angle)
        shear_matches = collect_angle_data(input_xy_exp, shear_data, angle)
        if not normal_matches:
            for row_idx in range(2):
                axes[row_idx, col_idx].axis("off")
            continue

        all_normal_load = np.concatenate([item[1] for item in normal_matches])
        all_normal_extension = np.concatenate([item[2] for item in normal_matches])
        beta_v = fit_origin_slope(all_normal_load, all_normal_extension)
        if angle == direct_shear_angle:
            beta_h_pred = np.nan
        else:
            beta_h_pred = 2.0 * beta_v * np.tan(np.deg2rad(angle))
        max_load = max(
            np.max(all_normal_load),
            max(np.max(item[1]) for item in shear_matches) if shear_matches else 0.0,
        )
        load_grid = np.linspace(0, max_load, 100)

        ax_normal = axes[0, col_idx]
        for exp_idx, load, extension in normal_matches:
            ax_normal.scatter(
                load,
                extension,
                s=18,
                alpha=0.7,
                edgecolor="none",
                label=f"Exp {exp_idx + 1}",
            )
        ax_normal.plot(
            load_grid,
            beta_v * load_grid,
            color="black",
            linewidth=2.2,
            label="Fit from normal data",
        )
        ax_normal.set_title(f"{angle} deg normal")
        ax_normal.set_ylabel("Normal extension [mm]")
        ax_normal.text(
            0.04,
            0.96,
            rf"$\hat{{\beta}}_v = {beta_v:.5g}$ mm/kN",
            transform=ax_normal.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        ax_shear = axes[1, col_idx]
        for exp_idx, load, extension in shear_matches:
            ax_shear.scatter(
                load,
                extension,
                s=18,
                alpha=0.7,
                edgecolor="none",
                label=f"Exp {exp_idx + 1}",
            )
        if shear_matches:
            all_shear_load = np.concatenate([item[1] for item in shear_matches])
            all_shear_extension = np.concatenate([item[2] for item in shear_matches])
            beta_h_observed = fit_origin_slope(all_shear_load, all_shear_extension)
        else:
            all_shear_load = np.array([])
            all_shear_extension = np.array([])
            beta_h_observed = np.nan

        shear_line_beta = beta_h_observed if angle == direct_shear_angle else beta_h_pred
        shear_line_label = (
            "Direct least-squares fit"
            if angle == direct_shear_angle
            else r"Prediction from normal fit"
        )
        ax_shear.plot(
            load_grid,
            shear_line_beta * load_grid,
            color="black",
            linewidth=2.2,
            label=shear_line_label,
        )
        ax_shear.set_title(f"{angle} deg shear")
        ax_shear.set_xlabel("Load [kN]")
        ax_shear.set_ylabel("Shear extension [mm]")
        if angle == direct_shear_angle:
            shear_text = rf"$\hat{{\beta}}_h = {beta_h_observed:.5g}$ mm/kN"
        else:
            shear_text = rf"$\beta_h^{{pred}} = 2\hat{{\beta}}_v\tan({angle}^\circ) = {beta_h_pred:.5g}$ mm/kN"
        ax_shear.text(
            0.04,
            0.96,
            shear_text,
            transform=ax_shear.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

        for ax in [ax_normal, ax_shear]:
            ax.axhline(0, color="0.75", linewidth=0.8)
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7, loc="best")

        rows.append(
            {
                "angle_deg": angle,
                "direction": "shear",
                "fit_type": (
                    "direct_origin_least_squares"
                    if angle == direct_shear_angle
                    else "predicted_from_normal_fit"
                ),
                "normal_slope_beta_v_mm_per_kN": float(beta_v),
                "predicted_shear_slope_beta_h_mm_per_kN": float(beta_h_pred),
                "observed_shear_slope_for_check_only_mm_per_kN": float(beta_h_observed),
                "observed_shear_over_predicted_shear": (
                    np.nan
                    if np.isnan(beta_h_pred)
                    else float(beta_h_observed / beta_h_pred)
                ),
                "n_normal_points": int(len(all_normal_load)),
                "n_shear_points": int(len(all_shear_load)),
                "n_normal_experiments": int(len(normal_matches)),
                "n_shear_experiments": int(len(shear_matches)),
            }
        )

    fig.suptitle("Normal-only origin fit with shear predicted by engineering-shear factor")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    plot_path = output_dir / "normal_fit_shear_factor_check.png"
    csv_path = output_dir / "normal_fit_shear_factor_check.csv"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(csv_path, index=False)
    print(f"Saved plot to {plot_path}")
    print(f"Saved slope check to {csv_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

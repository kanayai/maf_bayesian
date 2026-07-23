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
from src.vis.plotting import direction_display_label


def fit_origin_slope(load, extension):
    """Least-squares fit for extension = beta * load through the origin."""
    denominator = np.sum(load**2)
    if denominator <= 0:
        return np.nan
    return np.sum(load * extension) / denominator


def plot_extension_ratios(data, output_dir):
    input_xy_exp = data["input_xy_exp"]
    data_exp_v = data["data_exp_v"]
    data_exp_h = data["data_exp_h"]
    angles = [45, 135]
    eps = 1e-8

    fig, axes = plt.subplots(1, len(angles), figsize=(15, 4.5), sharey=False)
    ratio_rows = []

    for col_idx, angle in enumerate(angles):
        ax = axes[col_idx]
        angle_ratios = []
        angle_loads = []

        for exp_idx, xy in enumerate(input_xy_exp):
            exp_angle = int(round(np.rad2deg(float(xy[0, 1]))))
            if exp_angle != angle:
                continue

            load = np.asarray(xy[:, 0], dtype=float)
            ext_v = np.asarray(data_exp_v[exp_idx], dtype=float)
            ext_h = np.asarray(data_exp_h[exp_idx], dtype=float)

            valid = np.abs(ext_v) > eps
            ratio = ext_h[valid] / ext_v[valid]
            ratio_load = load[valid]

            if len(ratio) == 0:
                continue

            angle_ratios.append(ratio)
            angle_loads.append(ratio_load)
            ax.plot(
                ratio_load,
                ratio,
                marker="o",
                linestyle="-",
                markersize=3,
                linewidth=1,
                alpha=0.75,
                label=f"Exp {exp_idx + 1}",
            )

        if angle_ratios:
            all_ratios = np.concatenate(angle_ratios)
            all_loads = np.concatenate(angle_loads)
            finite = np.isfinite(all_ratios)
            all_ratios = all_ratios[finite]
            all_loads = all_loads[finite]

            median_ratio = float(np.median(all_ratios))
            mean_ratio = float(np.mean(all_ratios))
            ax.axhline(
                median_ratio,
                color="black",
                linewidth=2,
                label=f"Median = {median_ratio:.3g}",
            )
            reference_ratio = 2.0 if angle == 45 else -2.0
            ax.axhline(
                reference_ratio,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Reference = {reference_ratio:g}",
            )
            ax.text(
                0.04,
                0.96,
                f"median h/v = {median_ratio:.3g}\nmean h/v = {mean_ratio:.3g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ratio_rows.append(
                {
                    "angle_deg": angle,
                    "n_points": int(len(all_ratios)),
                    "ratio": "shear_h_extension / normal_v_extension",
                    "mean_ratio": mean_ratio,
                    "median_ratio": median_ratio,
                    "std_ratio": float(np.std(all_ratios)),
                    "min_ratio": float(np.min(all_ratios)),
                    "max_ratio": float(np.max(all_ratios)),
                }
            )
        else:
            ax.text(0.5, 0.5, "No valid ratios", transform=ax.transAxes, ha="center")

        ax.set_title(f"{angle} deg")
        ax.set_xlabel("Load [kN]")
        if col_idx == 0:
            ax.set_ylabel("Shear extension / normal extension")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Averaged extension ratios by loading angle")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    plot_path = output_dir / "empirical_extension_ratio_by_angle.png"
    csv_path = output_dir / "empirical_extension_ratio_by_angle.csv"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(csv_path, index=False)
    print(f"Saved extension ratio plot to {plot_path}")
    print(f"Saved extension ratio summaries to {csv_path}")
    print(ratio_df.to_string(index=False))


def main():
    data = load_all_data(config)
    input_xy_exp = data["input_xy_exp"]
    directions = {
        "v": data["data_exp_v"],
        "h": data["data_exp_h"],
    }
    angles_to_plot = [45, 135]

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(
        len(directions),
        len(angles_to_plot),
        figsize=(10, 7),
        squeeze=False,
        sharey=True,
    )

    for row_idx, direction in enumerate(["v", "h"]):
        dir_label = direction_display_label(direction)
        for col_idx, angle in enumerate(angles_to_plot):
            ax = axes[row_idx, col_idx]
            matching = []
            for exp_idx, xy in enumerate(input_xy_exp):
                exp_angle = int(round(np.rad2deg(float(xy[0, 1]))))
                if exp_angle == angle:
                    load = np.asarray(xy[:, 0], dtype=float)
                    extension = np.asarray(directions[direction][exp_idx], dtype=float)
                    matching.append((exp_idx, load, extension))

            if not matching:
                ax.axis("off")
                continue

            all_load = np.concatenate([item[1] for item in matching])
            all_extension = np.concatenate([item[2] for item in matching])
            beta = fit_origin_slope(all_load, all_extension)

            for exp_idx, load, extension in matching:
                ax.scatter(
                    extension,
                    load,
                    s=16,
                    alpha=0.65,
                    edgecolor="none",
                    label=f"Exp {exp_idx + 1}",
                )

            load_grid = np.linspace(0, np.max(all_load), 100)
            ax.plot(
                beta * load_grid,
                load_grid,
                color="black",
                linewidth=2,
                label="Origin LS fit",
            )

            rows.append(
                {
                    "direction": direction,
                    "direction_label": dir_label,
                    "angle_deg": angle,
                    "n_points": int(len(all_load)),
                    "n_experiments": int(len(matching)),
                    "slope_beta_mm_per_kN": float(beta),
                }
            )

            ax.set_title(f"{angle} deg {dir_label}")
            ax.set_xlabel("Averaged extension [mm]")
            if col_idx == 0:
                ax.set_ylabel("Load [kN]")
            ax.text(
                0.04,
                0.96,
                rf"$\hat{{\beta}} = {beta:.5g}$ mm/kN",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Origin-constrained empirical slopes from averaged experimental data")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    plot_path = output_dir / "empirical_origin_slope_estimates.png"
    csv_path = output_dir / "empirical_origin_slope_estimates.csv"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved plot to {plot_path}")
    print(f"Saved slope estimates to {csv_path}")
    print(pd.DataFrame(rows).to_string(index=False))

    plot_extension_ratios(data, output_dir)


if __name__ == "__main__":
    main()

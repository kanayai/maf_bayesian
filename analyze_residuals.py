
import arviz as az
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd

from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, posterior_predict

def main():
    print("Starting Residual Analysis...")
    
    # 1. Load Data
    data_dict = load_all_data(config)
    input_xy_exp = data_dict["input_xy_exp"]  # List of arrays
    input_xy_sim = data_dict["input_xy_sim"]
    input_theta_sim = data_dict["input_theta_sim"]
    data_exp_h = data_dict["data_exp_h"]
    data_exp_v = data_dict["data_exp_v"]
    data_sim_h = data_dict["data_sim_h"]
    data_sim_v = data_dict["data_sim_v"]
    
    # 2. Load Results
    results_dir = Path("results")
    files = list(results_dir.glob("*.nc"))
    if not files:
        print("No result files found in results/")
        return
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from {latest_file}")
    idata = az.from_netcdf(latest_file)
    
    # 3. Extract Samples (Subset)
    num_samples = 500
    print(f"Extracting {num_samples} posterior samples...")
    
    # Flatten posterior
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    total_samples = posterior.dims["sample"]
    indices = np.random.choice(total_samples, num_samples, replace=False)
    
    def get_batch(key):
        val = posterior[key].values
        # After stack, dim 0 is sample
        return jnp.array(val[indices])

    # Physical
    E_1 = get_batch("E_1")
    E_2 = get_batch("E_2")
    v_12 = get_batch("v_12")
    v_23 = get_batch("v_23")
    G_12 = get_batch("G_12")
    test_theta = jnp.stack([E_1, E_2, v_12, v_23, G_12], axis=1)

    # Hyper
    mu_emulator = get_batch("mu_emulator")
    sigma_emulator = get_batch("sigma_emulator")
    sigma_measure = get_batch("sigma_measure")
    
    # Lengths
    l_P = get_batch("lambda_P")
    l_alpha = get_batch("lambda_alpha")
    l_E1 = get_batch("lambda_E1")
    l_E2 = get_batch("lambda_E2")
    l_v12 = get_batch("lambda_v12")
    l_v23 = get_batch("lambda_v23")
    l_G12 = get_batch("lambda_G12")
    
    length_xy = jnp.stack([l_P, l_alpha], axis=1)
    length_theta = jnp.stack([l_E1, l_E2, l_v12, l_v23, l_G12], axis=1)

    # Helper to predict
    def predict_point(m_em, s_em, l_xy, l_th, s_meas, t_theta, t_xy, direction):
        full_input_xy_exp = jnp.concatenate(input_xy_exp, axis=0) # (Total_Exp_Points, 2)
        full_input_theta_exp = jnp.tile(t_theta, (full_input_xy_exp.shape[0], 1))
        
        full_data_exp = jnp.concatenate(data_exp_v, axis=0) if direction == 'v' else jnp.concatenate(data_exp_h, axis=0)
        trg_data_sim = data_sim_v if direction == 'v' else data_sim_h
        
        mean, std_em, _ = posterior_predict(
            random.PRNGKey(0), 
            full_input_xy_exp, input_xy_sim, full_input_theta_exp, input_theta_sim,
            full_data_exp, trg_data_sim,
            t_xy, t_theta, 
            m_em, s_em, l_xy, l_th, s_meas, direction=direction
        )
        return mean, std_em

    vmap_predict = vmap(predict_point, in_axes=(0,0,0,0,0,0,None,None))

    # Storage
    rows = []
    bands_data = [] # Store dense predictions for bands

    # Loop over experiments (different angles)
    for i, xy_exp in enumerate(input_xy_exp):
        angle_rad = xy_exp[0, 1]
        angle_deg = int(round(np.rad2deg(angle_rad)))
        print(f"  Processing Experiment {i+1} (Angle {angle_deg})...")
        
        loads = xy_exp[:, 0]
        max_load = jnp.max(loads)
        
        # --- 4a. Observed Points (Residuals) ---
        for direction, label in [('h', 'Shear'), ('v', 'Normal')]:
            # Predict at observed points
            means, stds_em = vmap_predict(
                mu_emulator, sigma_emulator, length_xy, length_theta, sigma_measure,
                test_theta, xy_exp, direction
            )
            
            mu_post = jnp.mean(means, axis=0) 
            
            # Total Uncertainty at Observed Points
            noise_var_samples = sigma_measure[:, None]**2 * loads[None, :] 
            gp_var_samples = stds_em**2
            total_std = jnp.sqrt(jnp.mean(gp_var_samples + noise_var_samples, axis=0) + jnp.var(means, axis=0))
            
            y_obs = (data_exp_h[i] if direction == 'h' else data_exp_v[i]).flatten()
            resid = y_obs - mu_post
            std_resid = resid / total_std
            
            for k in range(len(loads)):
                rows.append({
                    "Angle": angle_deg,
                    "Direction": label,
                    "Load": float(loads[k]),
                    "Residual": float(resid[k]),
                    "StdResidual": float(std_resid[k])
                })

            # --- 4b. Dense Grid (Smooth Bands) ---
            # Create dense test points
            dense_loads = jnp.linspace(0, max_load, 100)
            dense_angles = jnp.ones_like(dense_loads) * angle_rad
            dense_xy = jnp.stack([dense_loads, dense_angles], axis=1)
            
            # Predict on dense grid
            d_means, d_stds_em = vmap_predict(
                mu_emulator, sigma_emulator, length_xy, length_theta, sigma_measure,
                test_theta, dense_xy, direction
            )
            
            # Total Uncertainty on Dense Grid
            d_noise_var = sigma_measure[:, None]**2 * dense_loads[None, :]
            d_gp_var = d_stds_em**2
            d_total_std = jnp.sqrt(jnp.mean(d_gp_var + d_noise_var, axis=0) + jnp.var(d_means, axis=0))
            
            bands_data.append({
                "Angle": angle_deg,
                "Direction": label,
                "Load": dense_loads,
                "SigmaTotal": d_total_std
            })

    df = pd.DataFrame(rows)
    
    # 5. Plotting
    # New Output Structure: figures/<result_name>/
    result_name = latest_file.stem
    figures_dir = Path("figures") / result_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {figures_dir}")
    
    unique_combos = df[["Angle", "Direction"]].drop_duplicates().values
    
    # --- Plot: Residuals with Smooth Bands ---
    plt.figure(figsize=(15, 5 * len(unique_combos) // 3 + 5))
    num_plots = len(unique_combos)
    cols = 3
    rows_plt = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows_plt, cols, figsize=(15, 4*rows_plt), constrained_layout=True)
    if num_plots > 1: axes = axes.flatten()
    else: axes = [axes]
    
    for idx, (angle, direction) in enumerate(unique_combos):
        ax = axes[idx]
        
        # 1. Plot Residuals (Points)
        subset = df[(df["Angle"] == angle) & (df["Direction"] == direction)]
        ax.scatter(subset["Load"], subset["Residual"], alpha=0.6, label="Residuals")
        
        # 2. Plot Smooth Bands using dense data
        # Find corresponding band data
        band = next(b for b in bands_data if b["Angle"] == angle and b["Direction"] == direction)
        
        # 95% CI = +/- 1.96 * Sigma
        upper = 1.96 * band["SigmaTotal"]
        lower = -1.96 * band["SigmaTotal"]
        
        ax.fill_between(band["Load"], lower, upper, color='gray', alpha=0.2, label="95% Prediction Interval")
        ax.plot(band["Load"], upper, color='gray', linestyle='--', alpha=0.5)
        ax.plot(band["Load"], lower, color='gray', linestyle='--', alpha=0.5)
        
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_title(f"Angle {angle}° - {direction}")
        ax.set_xlabel("Load [kN]")
        ax.set_ylabel("Residual [mm]")
        if idx == 0: ax.legend()
            
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
        
    plt.savefig(figures_dir / "residuals_with_bands.png")
    
    # --- Other Plots ---
    # Standardized Residuals vs Load
    g = sns.FacetGrid(df, col="Angle", row="Direction", sharex=False, sharey=True, height=4, aspect=1.2)
    g.map(plt.scatter, "Load", "StdResidual", alpha=0.6)
    for ax in g.axes.flat:
        ax.axhline(0, color='k', ls='-', lw=1)
        ax.axhline(2, color='r', ls='--', lw=1)
        ax.axhline(-2, color='r', ls='--', lw=1)
    g.set_axis_labels("Load [kN]", "Standardized Residual")
    g.savefig(figures_dir / "std_residuals_vs_load.png")

    # --- Histogram of Standardized Residuals ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["StdResidual"], kde=True, stat="density", label="Observed")
    
    # Overlay Standard Normal
    x_range = np.linspace(df["StdResidual"].min()-1, df["StdResidual"].max()+1, 100)
    from scipy.stats import norm
    plt.plot(x_range, norm.pdf(x_range, 0, 1), 'r--', lw=2, label="Standard Normal")
    
    plt.title("Distribution of Standardized Residuals")
    plt.xlabel("Standardized Residual")
    plt.legend()
    plt.savefig(figures_dir / "std_residuals_hist.png")

    # --- Q-Q Plot ---
    plt.figure(figsize=(8, 8))
    import scipy.stats as stats
    stats.probplot(df["StdResidual"], dist="norm", plot=plt)
    plt.title("Q-Q Plot of Standardized Residuals")
    plt.savefig(figures_dir / "residuals_qq.png")

    print("Done.")

if __name__ == "__main__":
    main()

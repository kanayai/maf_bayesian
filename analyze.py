print("DEBUG: Pre-Import")
import matplotlib
matplotlib.use("Agg")
import arviz as az
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import jax
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from numpyro.infer import Predictive
import pandas as pd
import seaborn as sns
from datetime import datetime
import argparse


from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, model_empirical, model_simple, posterior_predict
from src.io.output_manager import save_config_log
from src.vis.plotting import (
    plot_experimental_data,
    plot_averaged_experimental_data,
    plot_posterior_distributions,
    plot_prediction,
    plot_combined_prediction,
    plot_grid_prediction,
    plot_spaghetti_verification,
    plot_grid_spaghetti,
    plot_distributions_grid_2x3,
    plot_bias_column_layout,
)

def predict_sample_wrapper(
    rng,
    m_em,
    s_em,
    l_xy,
    l_th,
    s_meas,
    s_meas_base,
    s_const,
    bias_slope_item,
    t_theta,
    t_xy,
    exp_xy,
    input_xy_sim,
    input_theta_sim,
    exp_data, # (N, 3) or (N,)
    sim_data,
    direction,
    is_prior,
    g_scale_v,
    g_scale_h,
    betas_simp,
    sigma_simp,
    noise_model
):
    """
    Module-level wrapper for single sample prediction.
    Designed to be JIT-compiled via vmap.
    """
    # Construct exp_theta if posterior
    if is_prior:
         exp_theta = jnp.empty((0, 5))
    else:
         exp_theta = jnp.tile(t_theta, (exp_xy.shape[0], 1))

    # Call posterior_predict
    mean, std_em, sample = posterior_predict(
        rng,
        exp_xy,
        input_xy_sim,
        exp_theta,
        input_theta_sim,
        exp_data,
        sim_data,
        t_xy,
        t_theta,
        m_em,
        s_em,
        l_xy,
        l_th,
        s_meas,
        s_meas_base,
        s_const,
        direction=direction,
        bias_slope=bias_slope_item,
        gamma_scale_v=g_scale_v,
        gamma_scale_h=g_scale_h,
        betas_simple=betas_simp,
        sigma_simple=sigma_simp,
    )
    return mean, std_em, sample

# Global JIT-compiled prediction function
# Global JIT-compiled prediction function
# Global JIT-compiled prediction function
# Global JIT-compiled prediction function
predict_fn = jax.jit(
    vmap(
        predict_sample_wrapper, 
        in_axes=(
            0, # rng (1)
            0, # m_em
            0, # s_em
            0, # l_xy
            0, # l_th
            0, # s_meas
            0, # s_meas_base
            0, # s_const
            0, # bias_slope_item
            0, # t_theta (10)
            None, # t_xy (11)
            None, # exp_xy
            None, # input_xy_sim
            None, # input_theta_sim
            None, # exp_data
            None, # sim_data
            None, # direction
            None, # is_prior (18)
            0, # g_scale_v
            0, # g_scale_h
            0, # betas_simp
            0, # sigma_simp
            None # noise_model (23)
        )
    ), 
    static_argnames=("direction", "is_prior", "noise_model")
)


def main():
    print("DEBUG: Script Start")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze MCMC results and generate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                  # Default: saves to figures/analysis_<timestamp>/
  python analyze.py --experimental   # Saves to figures/tmp/analysis_<timestamp>/
  python analyze.py --final          # Saves to figures/final/analysis_<timestamp>/
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--experimental",
        action="store_true",
        help="Save figures to figures/tmp/ (for experimental/testing runs)",
    )
    mode_group.add_argument(
        "--final",
        action="store_true",
        help="Save figures to figures/final/ (for important/final runs)",
    )

    args = parser.parse_args()

    # Determine output mode
    if args.experimental:
        output_mode = "experimental"
        print("🧪 Running in EXPERIMENTAL mode - figures will be saved to figures/tmp/")
    elif args.final:
        output_mode = "final"
        print("📌 Running in FINAL mode - figures will be saved to figures/final/")
    else:
        output_mode = "default"
        print("Running in default mode - figures will be saved to figures/")

    print("Starting analysis...")

    # 1. Load Data
    data_dict = load_all_data(config)
    print(f"DEBUG: Loaded data_dict keys: {list(data_dict.keys())}")

    # 2. Plot Experimental Data
    # Setup output directory
    model_type = config.get("model_type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = timestamp

    # Determine base figures directory based on mode
    if output_mode == "experimental":
        base_dir = Path("figures") / "tmp"
    elif output_mode == "final":
        base_dir = Path("figures") / "final"
    else:
        base_dir = Path("figures")

    figures_dir = base_dir / f"analysis_{model_type}_{timestamp}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {figures_dir}")

    # Save Config Log (Output Rules) - Done early to ensure it exists even if later steps fail
    # We don't have the results filename yet if we haven't loaded it, but we should load it first?
    # Actually step 3 loads it. Let's move this call after step 3.

    # Create temporary dict for full data plotting
    # The plotting functions expect standard keys, so we map the full data to them
    full_data_dict = {
        "input_xy_exp": data_dict["input_xy_exp_full"],
        "data_exp_h_raw": data_dict["data_exp_h_full_raw"],
        "data_exp_v_raw": data_dict["data_exp_v_full_raw"],
    }

    plot_experimental_data(
        full_data_dict, save_path=figures_dir / "experimental_data.png"
    )

    plot_averaged_experimental_data(
        full_data_dict, save_path=figures_dir / "experimental_data_averaged.png"
    )

    # 3. Load Latest Results
    # 3. Load Latest Results
    if output_mode == "experimental":
        results_dir = Path("results") / "tmp"
    elif output_mode == "final":
        results_dir = Path("results") / "final"
    else:
        results_dir = Path("results")
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist.")
        return

    files = list(results_dir.glob("*.nc"))
    if not files:
        print("No result files found in results/")
        return

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from {latest_file}")
    idata = az.from_netcdf(latest_file)

    # Save Config Log
    save_config_log(config, figures_dir, latest_file.name)

    # 4. Categorized Posterior Plots
    samples = {}
    for key in idata.posterior.data_vars:
        data = idata.posterior[key].values
        flat_data = data.reshape(-1, *data.shape[2:])
        samples[key] = jnp.array(flat_data)

    # Define categories
    physical_params = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    hyper_params = [
        "mu_emulator_v",
        "mu_emulator_h",
        "sigma_emulator",
        "sigma_measure",
        "sigma_measure_base",
        "sigma_constant",
        "sigma_b_slope",
        "gamma_scale_v",
        "gamma_scale_h",
        "lambda_P",
        "lambda_alpha",
        "lambda_E1",
        "lambda_E2",
        "lambda_v12",
        "lambda_v23",
        "lambda_G12",
    ]

    # Filter samples into categories
    # Also filter out parameters that are all zeros (unused in current noise model)
    samples_physical = {k: v for k, v in samples.items() if k in physical_params}
    samples_hyper = {
        k: v
        for k, v in samples.items()
        if k in hyper_params and jnp.any(v != 0)  # Exclude all-zero parameters
    }
    samples_bias = {
        k: v
        for k, v in samples.items()
        if (k.startswith("b_") or (k.startswith("sigma_b") and k != "sigma_b_slope")) and not k.endswith("_n")
    }
    # Separate _n parameters (normalized)
    samples_n = {k: v for k, v in samples.items() if k.endswith("_n")}
    # Gamma factors (Empirical Model) - Exclude hyper parameters like "gamma_scale"
    samples_gamma = {
        k: v for k, v in samples.items() 
        if k.startswith("gamma_") and not k.startswith("gamma_scale")
    }

    # Unpack data
    input_xy_exp = data_dict["input_xy_exp"]
    input_xy_sim = data_dict["input_xy_sim"]
    print(f"DEBUG_ANALYZE: input_xy_sim shape: {input_xy_sim.shape}")
    print(f"DEBUG_ANALYZE: input_xy_sim head:\n{input_xy_sim[:5]}")
    input_theta_sim = data_dict["input_theta_sim"]
    data_exp_h = data_dict["data_exp_h"]
    data_exp_v = data_dict["data_exp_v"]
    data_sim_h = data_dict["data_sim_h"]
    data_sim_v = data_dict["data_sim_v"]

    # --- Generate Prior Samples (Moved before plotting to allow prior plots) ---
    print("Generating prior samples...")
    num_pred_samples = config["data"].get("prediction_samples", 500)
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)

    # We need to run Predictive to get priors for all variables
    # Sort out model arguments based on type
    # Pre-calculate angle indices map (Required by updated model_empirical and model_simple)
    standard_angles = [45, 90, 135]
    exp_angle_indices = []
    # data_dict["input_xy_exp"] is list of exp arrays
    for i in range(len(data_dict["input_xy_exp"])):
        ang_rad = data_dict["input_xy_exp"][i][0, 1]
        ang_deg = int(round(jnp.degrees(ang_rad)))
        try:
            idx = standard_angles.index(ang_deg)
            exp_angle_indices.append(idx)
        except ValueError:
            exp_angle_indices.append(-1)
            
    if model_type == "model_empirical":
                
        model_func = model_empirical
        model_args = (
            data_dict["input_xy_exp"],
            data_dict["data_exp_h_raw"],
            data_dict["data_exp_v_raw"],
            jnp.array(exp_angle_indices), # Pass angle indices
            config,
        )
    elif model_type == "model_simple":
         model_func = model_simple
         model_args = (
             data_dict["input_xy_exp"],
             data_dict["data_exp_h_raw"], # pass raw or list? model_simple expects list of arrays matching input_xy_exp
             data_dict["data_exp_v_raw"],
             jnp.array(exp_angle_indices), # Pass angle indices
             config
         )
    else:
        model_func = model_n_hv
        model_args = (
            input_xy_exp,
            input_xy_sim,
            input_theta_sim,
            data_exp_h,
            data_exp_v,
            data_sim_h,
            data_sim_v,
            config,
        )

    prior_predictive = Predictive(model_func, num_samples=num_pred_samples)
    prior_samples_all = prior_predictive(
        rng_key_prior,
        *model_args
    )

    # --- Analytical Priors ---
    # We construct functions that return the log_prob (or pdf) for plotting
    # For reparameterized priors (model_n_hv), the physical params E_1 etc are:
    # val = mean + scale * N(0,1) -> val ~ N(mean, scale)

    from scipy.stats import norm, expon, lognorm, lognorm

    priors_config = config["priors"]

    def get_prior_pdf(key, x_vals):
        # Physical parameters (Theta) - Reparameterized
        if key in priors_config["theta"]["reparam"]:
            p = priors_config["theta"]["reparam"][key]
            return norm.pdf(x_vals, loc=p["mean"], scale=p["scale"])

        # Hyperparameters
        # Length scales: val = exp(mean + scale * N(0,1)) -> LogNormal
        # But wait, the samples we have for lambda_* are the raw N(0,1) samples?
        # No, let's check models.py.
        # lambda_P = numpyro.sample("lambda_P", dist.Normal(0, 1))
        # length_P = jnp.exp(mu + sigma * lambda_P)
        # The posterior samples contain "lambda_P" (the N(0,1) variable) AND potentially the transformed one if deterministic?
        # The user asked for plots of "lambda_P". In the model, "lambda_P" is Standard Normal N(0,1).
        # So for any parameter that is sampled as N(0,1) in the model (like E_1_n, lambda_P, etc.), the prior is N(0,1).

        # Let's check what keys we are plotting.
        # Physical: E_1, E_2...
        # In model_n_hv: E_1 = E_1_mean + E_1_scale * E_1_n
        # E_1 is a Deterministic node.
        # So its prior is N(E_1_mean, E_1_scale).
        if key in priors_config["theta"]["reparam"]:
            p = priors_config["theta"]["reparam"][key]
            return norm.pdf(x_vals, loc=p["mean"], scale=p["scale"])

        # Hyperparameters (Lambdas) - LogNormal reparameterization
        if key.startswith("lambda_"):
            # lambda ~ LogNormal via exp(log_mean + log_scale * N(0,1))
            # In scipy.stats.lognorm(s, scale): s=log_scale, scale=exp(log_mean)
            if key in priors_config["hyper"]["length_scales"]:
                p = priors_config["hyper"]["length_scales"][key]
                s = p["log_scale"]
                scale = np.exp(p["log_mean"])
                return lognorm.pdf(x_vals, s=s, scale=scale)
            return None

        # Emulator Mean/Scale
        # mu_emulator: Normal(mean, scale) or LogNormal(log_mean, log_scale)
        if key in ["mu_emulator_v", "mu_emulator_h"]:
            if key in priors_config["hyper"]:
                 p = priors_config["hyper"][key]
                 if "log_mean" in p:
                     return lognorm.pdf(x_vals, s=p["log_scale"], scale=np.exp(p["log_mean"]))
                 return norm.pdf(x_vals, loc=p["mean"], scale=p["scale"])
            if "log_mean" in p:
                # LogNormal: val = exp(log_mean + log_scale * N(0,1))
                # scipy lognorm: s=log_scale, scale=exp(log_mean)
                return lognorm.pdf(x_vals, s=p["log_scale"], scale=np.exp(p["log_mean"]))
            else:
                return norm.pdf(x_vals, loc=p["mean"], scale=p["scale"])

        # Helper to extract PDF from numpyro distribution in config
        def get_pdf_from_target_dist(hyper_key):
            """Extract scipy PDF from numpyro target_dist in config."""
            from scipy.stats import truncnorm
            import numpyro.distributions as npdist
            
            if hyper_key not in priors_config["hyper"]:
                return None
            cfg = priors_config["hyper"][hyper_key]
            if "target_dist" not in cfg:
                return None
            
            d = cfg["target_dist"]
            dist_name = type(d).__name__
            
            # Handle different distribution types
            if isinstance(d, npdist.Exponential):
                rate = float(d.rate)
                return expon.pdf(x_vals, scale=1 / rate)
            elif isinstance(d, npdist.LogNormal):
                return lognorm.pdf(x_vals, s=d.scale, scale=np.exp(d.loc))
            elif dist_name == "TruncatedDistribution" or "Truncated" in dist_name:
                # TruncatedNormal creates a TruncatedDistribution object
                # Extract parameters from the base distribution
                try:
                    base = getattr(d, 'base_dist', d)  # Get base distribution
                    loc = float(base.loc) if hasattr(base, 'loc') else 0.0
                    scale = float(base.scale) if hasattr(base, 'scale') else 1.0
                    low = float(d.low) if hasattr(d, 'low') else -np.inf
                    high = float(d.high) if hasattr(d, 'high') else np.inf
                    # scipy truncnorm uses (a, b) = (low - loc) / scale, (high - loc) / scale
                    a = (low - loc) / scale if low != -np.inf else -np.inf
                    b = (high - loc) / scale if high != np.inf else np.inf
                    return truncnorm.pdf(x_vals, a, b, loc=loc, scale=scale)
                except Exception as e:
                    print(f"Warning: Could not extract TruncatedNormal params: {e}")
                    return None
            elif isinstance(d, npdist.Normal):
                loc = float(d.loc)
                scale = float(d.scale)
                return norm.pdf(x_vals, loc=loc, scale=scale)
            else:
                # Fallback: try to use log_prob if available
                try:
                    import jax.numpy as jnp
                    return np.exp(d.log_prob(jnp.array(x_vals)))
                except:
                    return None

        # Helper to get LogNormal PDF from log_mean/log_scale config
        def get_lognorm_pdf(hyper_key):
            """Get LogNormal PDF from log_mean/log_scale config."""
            if hyper_key not in priors_config["hyper"]:
                return None
            cfg = priors_config["hyper"][hyper_key]
            if "log_mean" not in cfg:
                return None
            # scipy lognorm: s=log_scale, scale=exp(log_mean)
            return lognorm.pdf(x_vals, s=cfg["log_scale"], scale=np.exp(cfg["log_mean"]))

        # Hyperparameters with target_dist or log_mean/log_scale
        if key == "sigma_emulator":
            cfg = priors_config["hyper"]["sigma_emulator"]
            if "log_mean" in cfg:
                return get_lognorm_pdf("sigma_emulator")
            return get_pdf_from_target_dist("sigma_emulator")
        if key == "sigma_measure":
            cfg = priors_config["hyper"]["sigma_measure"]
            if "log_mean" in cfg:
                return get_lognorm_pdf("sigma_measure")
            return get_pdf_from_target_dist("sigma_measure")
        if key == "sigma_measure_base":
            return get_pdf_from_target_dist("sigma_measure_base")
        if key == "sigma_constant":
            return get_pdf_from_target_dist("sigma_constant")

        # Bias
        if key.startswith("b_"):
            # b_E1 ~ Normal(0, sigma_b_E1)
            # This is hierarchical. We can't easily plot a marginal prior for b_E1 without integrating out sigma_b_E1.
            # Or maybe sigma_b_E1 is fixed? No, it has a prior.
            # For now, maybe skip prior for bias terms or just plot N(0, 1) if they were normalized?
            # They are: b_E1 = sigma_b_E1 * b_E1_n. b_E1_n ~ N(0,1).
            return None

        if key.startswith("sigma_b"):
             # bias_priors in config
             bias_priors = priors_config.get("bias_priors", {})
             if key in bias_priors:
                 d = bias_priors[key]
                 # If it is a distribution object
                 import numpyro.distributions as npdist
                 if isinstance(d, npdist.Exponential):
                     rate = float(d.rate)
                     return expon.pdf(x_vals, scale=1 / rate)
             
             # Fallback for old/hardcoded
             if "E1" in key:
                return expon.pdf(x_vals, scale=1 / 0.001) # 0.001 from old default? or 0.0001?
             if "alpha" in key:
                return expon.pdf(
                    x_vals, scale=np.deg2rad(10)
                ) 


        # Normalized params (_n)
        if key.startswith("lambda_"):
            if key in priors_config["hyper"]["length_scales"]:
                p = priors_config["hyper"]["length_scales"][key]
                s = p["scale"]
                scale = np.exp(p["mean"])
                return lognorm.pdf(x_vals, s=s, scale=scale)
            return norm.pdf(x_vals, loc=0.0, scale=1.0)

        # Gamma factors (Empirical Model)
        if key.startswith("gamma_"):
            # Handle gamma_scale_v and gamma_scale_h which are now Exponential(100) -> Mean=0.01
            # They start with gamma_ but are hyperparameters, not angle-specific factors
            if key in ["gamma_scale_v", "gamma_scale_h"]:
                 # Extract from config["priors"]["hyper"][key]["target_dist"]
                 if key in priors_config["hyper"]:
                     d = priors_config["hyper"][key].get("target_dist")
                     if hasattr(d, "rate"):
                         rate = float(d.rate)
                         return expon.pdf(x_vals, scale=1 / rate)
                 
                 # Fallback if not found or no rate
                 rate = 10.0
                 return expon.pdf(x_vals, scale=1 / rate)

            try:
                # key format: "gamma_v_45" or "gamma_h_135"
                parts = key.split("_")
                # parts[0]="gamma", parts[1]="v"/"h", parts[2]=angle_deg
                direction = parts[1]
                angle_deg = int(parts[2])
                angle_rad = np.deg2rad(angle_deg)
                
                # Get priors from samples if possible? 
                # For plotting prior *density*, we usually use the configured mean/std.
                # Now that gamma_scale is inferred, we should technically integrate over the prior of gamma_scale?
                # Or just use the mean of the prior of gamma_scale (0.01) for visualization?
                # Using 0.01 is a reasonable approximation for "prior density of gamma given mean prior".
                
                empirical_config = config.get("empirical", {})
                gamma_scale_v = 0.01 
                gamma_scale_h = 0.01
                
                if direction == "v":
                    mu = jnp.cos(angle_rad)
                    gamma_scale = gamma_scale_v
                else:
                    mu = jnp.sin(angle_rad)
                    gamma_scale = gamma_scale_h
                    
                return norm.pdf(x_vals, loc=mu, scale=gamma_scale)
            except (ValueError, IndexError):
                return None

        return None

    # Plot categories
    import pandas as pd

    def save_stats_csv(samples_dict, filename):
        stats = []
        for key, val in samples_dict.items():
            # Calculate stats
            mean = jnp.mean(val)
            var = jnp.var(val)
            std = jnp.std(val)

            stats.append({"Parameter": key, "Mean": mean, "Variance": var, "Std": std})

        df = pd.DataFrame(stats)
        df.set_index("Parameter", inplace=True)
        df.to_csv(figures_dir / filename)
        print(f"Saved stats to {figures_dir / filename}")

    if model_type == "model_simple":
        # Simple model Analysis
        print("--- Running Simple Model Analysis ---")
        
        # 1. Plot Posteriors for Beta and Sigma
        # Define priors for plotting
        import numpyro.distributions as npdist

        def get_simple_prior(key, x_vals):
            # Read from config["priors"]["simple"]
            # We assume config is available (imported globally)
            priors_simple = config["priors"].get("simple", {})
            
            if key.startswith("beta_"):
                 if key in priors_simple:
                     entry = priors_simple[key]
                     # Check for standardized format first
                     if "log_mean" in entry:
                         return lognorm.pdf(x_vals, s=entry["log_scale"], scale=np.exp(entry["log_mean"]))
                     elif "target_dist" in entry:
                         d = entry["target_dist"]
                         if isinstance(d, npdist.Normal):
                             return norm.pdf(x_vals, loc=d.loc, scale=d.scale)
                         elif isinstance(d, npdist.LogNormal):
                             # Numpyro LogNormal(loc, scale) -> log(X) ~ N(loc, scale)
                             # Scipy lognorm(s=scale, scale=exp(loc))
                             return lognorm.pdf(x_vals, s=d.scale, scale=np.exp(d.loc))
                 # Fallback
                 return norm.pdf(x_vals, loc=0, scale=10)
                 
            if "sigma" in key:
                 # Unified sigma_measure is now in hyper
                 if "sigma_measure" in config["priors"]["hyper"]:
                      entry = config["priors"]["hyper"]["sigma_measure"]
                      if "log_mean" in entry:
                          return lognorm.pdf(x_vals, s=entry["log_scale"], scale=np.exp(entry["log_mean"]))
                      elif "target_dist" in entry:
                          d = entry["target_dist"]
                          if isinstance(d, npdist.Exponential):
                              return expon.pdf(x_vals, scale=1/d.rate)
                          elif isinstance(d, npdist.LogNormal):
                              return lognorm.pdf(x_vals, s=d.scale, scale=np.exp(d.loc))
                 # Fallback
                 return expon.pdf(x_vals, scale=0.1)
            return None

        # 1. Plot Posteriors for Beta (Grid Layout: Row 0=H, Row 1=V)
        # Order: h_45, h_90, h_135, v_45, v_90, v_135
        beta_params = [
            "beta_h_45", "beta_h_90", "beta_h_135",
            "beta_v_45", "beta_v_90", "beta_v_135"
        ]
        samples_betas = {k: samples[k] for k in beta_params if k in samples}
        
        # Plot Betas in 2x3 grid
        plot_posterior_distributions(
            samples_betas,
            prior_pdf_fn=get_simple_prior,
            save_path=figures_dir / f"posterior_simple_betas_{suffix}.png",
            layout_rows=2
        )
        save_stats_csv(samples_betas, f"inference_simple_beta_stats_{suffix}.csv")



        # 2. Plot Sigma separately
        if "sigma_measure" in samples:
             samples_sigma = {"sigma_measure": samples["sigma_measure"]}
             plot_posterior_distributions(
                 samples_sigma,
                 prior_pdf_fn=get_simple_prior,
                 save_path=figures_dir / f"posterior_simple_sigma_{suffix}.png"
             )
             save_stats_csv(samples_sigma, f"inference_simple_sigma_stats_{suffix}.csv")

        # 2. Plot Spaghetti (Linear Fits)
        # We want to plot the lines Y = P * beta vs Data
        # For each direction and angle (though angle doesn't affect beta here, just data splitting)
        
        print("Generating simple linear plots...")
        # Get samples reference for indexing
        # Any beta will do
        ref_beta = samples["beta_v_45"]
        
        # Subset for plotting (e.g. 100 samples)
        n_plot = min(100, len(ref_beta))
        idxs = np.random.choice(len(ref_beta), n_plot, replace=False)
        
        # Iterate over angles/directions to just show data vs fits
        # Since beta is global, the fit is the same for all angles, but we split plots by angle for clarity
        
        angles = [45, 90, 135]
        max_load = config["data"].get("max_load", 10.0)
        load_grid = np.linspace(0, max_load, 100)
        
        for angle in angles:
             fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
             
             # Filter Data for this angle
             # We need to find which experiments correspond to this angle
             # data_dict["input_xy_exp"] has info
             
             # Directions
             for i, direction in enumerate(["v", "h"]):
                 ax = axes[i]
                 
                 # Get specific beta for this direction and angle
                 beta_key = f"beta_{direction}_{angle}"
                 if beta_key not in samples:
                      print(f"Warning: {beta_key} not found in samples")
                      continue
                      
                 beta_full = samples[beta_key]
                 beta_samps = beta_full[idxs]
                 
                 # Plot Data
                 # Find experiments with this angle
                 ang_rad = np.deg2rad(angle)
                 # Tolerance for float comparison
                 
                 # We iterate through all experiments
                 for e_idx, xy in enumerate(data_dict["input_xy_exp"]):
                      # xy is (N, 2), check angle
                      if np.isclose(xy[0, 1], ang_rad, atol=0.1):
                           # Plot this experiment
                           # raw data
                           raw_data = data_dict["data_exp_v_raw"][e_idx] if direction == "v" else data_dict["data_exp_h_raw"][e_idx]
                           # raw_data is (N, 3)
                           load_exp = xy[:, 0]
                           ax.plot(load_exp, raw_data, "k.", alpha=0.1) # Plot all 3 reps
                           
                 # Plot Estimates
                 # Lines: Y = P * trig * beta
                 
                 ang_rad = np.deg2rad(angle)
                 if direction == "v":
                     trig_factor = np.cos(ang_rad)
                 else:
                     trig_factor = np.sin(ang_rad)
                     
                 for b in beta_samps:
                      ax.plot(load_grid, load_grid * trig_factor * b, "r-", alpha=0.1)
                      
                 # Mean estimate
                 b_mean = np.mean(beta_full)
                 ax.plot(load_grid, load_grid * trig_factor * b_mean, "r-", linewidth=2, label="Mean Fit")
                 
                 ax.set_title(f"Angle {angle}° - Direction {direction.upper()}")
                 ax.set_xlabel("Load [kN]")
                 if i == 0: ax.set_ylabel("Extension [mm]")
                 
             plt.tight_layout()
             plt.savefig(figures_dir / f"simple_fit_{angle}_{suffix}.png")
             plt.close()
             
        print("Simple analysis complete.")
        # Removed return so standard prediction plots are generated
        # return


    # (Function moved up)

    # Filename suffix
    suffix = timestamp

    # Enforce order for physical params
    # E_1, E_2, v_12, v_23, G_12
    ordered_physical = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    samples_physical_ordered = {
        k: samples_physical[k] for k in ordered_physical if k in samples_physical
    }

    if samples_physical_ordered:
        plot_posterior_distributions(
            samples_physical_ordered,
            prior_pdf_fn=get_prior_pdf,
            save_path=figures_dir / f"posterior_physical_{suffix}.png",
            layout_rows=1,
        )  # Force 1 row
        save_stats_csv(samples_physical_ordered, f"inference_theta_stats_{suffix}.csv")

    if samples_hyper:
        # Exclude sigma_measure if already plotted in simple analysis
        if config["model_type"] == "model_simple" and "sigma_measure" in samples_hyper:
            samples_hyper_plot = {k: v for k, v in samples_hyper.items() if k != "sigma_measure"}
        else:
            samples_hyper_plot = samples_hyper

        if samples_hyper_plot:
            # Organize hyperparameters in a 2x3 grid layout (transposed):
            # Row 1: gamma_scale_v, mu_emulator_v, sigma_b_slope
            # Row 2: gamma_scale_h, mu_emulator_h, sigma_measure
            hyper_order = [
                "gamma_scale_v", "mu_emulator_v", "sigma_b_slope",
                "gamma_scale_h", "mu_emulator_h", "sigma_measure"
            ]
            # Filter to only include params that exist in samples
            samples_hyper_ordered = {
                k: samples_hyper_plot[k] for k in hyper_order if k in samples_hyper_plot
            }
            # Add any remaining hyper params not in the order (in case some are added later)
            for k, v in samples_hyper_plot.items():
                if k not in samples_hyper_ordered:
                    samples_hyper_ordered[k] = v
            
            # Unified range for parameters that should be compared
            shared_xlim_groups = {
                "mu_emulator": ["mu_emulator_v", "mu_emulator_h"],
                "gamma_scale": ["gamma_scale_v", "gamma_scale_h"]
            }
            
            plot_posterior_distributions(
                samples_hyper_ordered,
                prior_pdf_fn=get_prior_pdf,
                save_path=figures_dir / f"posterior_hyper_{suffix}.png",
                layout_rows=2,  # Transposed: 2 rows x 3 cols
                shared_xlim_groups=shared_xlim_groups,
            )
            save_stats_csv(samples_hyper_ordered, f"inference_hyper_stats_{suffix}.csv")

    if samples_bias:
        print(f"DEBUG: samples_bias len: {len(samples_bias)}")
        print(f"DEBUG: samples_bias keys: {list(samples_bias.keys())[:5]}")
        # Check if we have "b_slope" array and remove it if individual keys exist
        if "b_slope" in samples_bias and len(samples_bias) > 1:
            if any(k.startswith("b_") and k != "b_slope" for k in samples_bias):
                samples_bias.pop("b_slope", None)
        
        # Rename keys: b_{i}_slope -> b_{angle}_{count}
        # First, map experiment index to angle
        exp_angles = []
        for i in range(len(data_dict["input_xy_exp"])):
            ang_rad = data_dict["input_xy_exp"][i][0, 1]
            ang_deg = int(round(np.rad2deg(ang_rad)))
            exp_angles.append(ang_deg)
            
        # Count occurrences to decide on numbering
        from collections import Counter
        angle_counts_total = Counter(exp_angles)
        angle_counters = {a: 0 for a in exp_angles}
        
        samples_bias_renamed = {}
        bias_key_map = {}
        
        for k, v in samples_bias.items():
            # semantic check: is this an indexed bias param?
            # distinct from sigma_b_slope
            if k.startswith("b_") and "_slope" in k and "sigma" not in k:
                # Try to parse index
                # Format: b_{i}_slope
                try:
                    # Extract i (1-based index)
                    prefix = k.split("_slope")[0] # "b_1"
                    idx_str = prefix.split("_")[1] # "1"
                    if idx_str.isdigit():
                        idx = int(idx_str) - 1 # 0-based
                        
                        if 0 <= idx < len(exp_angles):
                            ang = exp_angles[idx]
                            angle_counters[ang] += 1
                            count = angle_counters[ang]
                            
                            # Naming convention
                            if angle_counts_total[ang] > 1:
                                new_key = f"b_{ang}_{count}"
                            else:
                                new_key = f"b_{ang}"
                            
                            samples_bias_renamed[new_key] = v
                            bias_key_map[new_key] = k
                        else:
                            # Index out of range? Keep original
                            samples_bias_renamed[k] = v
                            bias_key_map[k] = k
                    else:
                        samples_bias_renamed[k] = v
                        bias_key_map[k] = k
                except Exception:
                     samples_bias_renamed[k] = v
                     bias_key_map[k] = k
            else:
                # Keep other params (like sigma_b_slope)
                samples_bias_renamed[k] = v
                bias_key_map[k] = k

        # User requested specific column layout: 45 | 90 | 135
        
        # Group samples by angle for this plotter
        bias_data_by_angle = {45: [], 90: [], 135: []}
        
        for k, v in samples_bias_renamed.items():
             # Parse b_{ang}_{count}
             try:
                 parts = k.split("_")
                 # parts[1] is angle
                 ang = int(parts[1])
                 if ang in bias_data_by_angle:
                     # Label: key is fine
                     # Pass original key for prior lookup
                     original_key = bias_key_map.get(k, k)
                     bias_data_by_angle[ang].append((k, v, original_key))
             except:
                 continue
                 
        print(f"DEBUG: bias_data_by_angle counts: { {k: len(v) for k,v in bias_data_by_angle.items()} }")

        plot_bias_column_layout(
            bias_data_by_angle,
            save_path=figures_dir / f"posterior_bias_columns_{suffix}.png",
            # No prior plotted: b_i ~ N(0, sigma_b) has hierarchical prior (sigma_b is inferred)
        )
        save_stats_csv(samples_bias_renamed, f"inference_bias_stats_{suffix}.csv")

    # Handle normalized parameters
    # 1. Move sigma_measure_n to hyper
    if "sigma_measure_n" in samples_n:
        samples_hyper["sigma_measure_n"] = samples_n.pop("sigma_measure_n")
        
    # 2. Extract and format b_i_slope_n for column layout
    samples_bias_n = {}
    keys_to_remove = []
    
    for k, v in samples_n.items():
        if k.startswith("b_") and "_slope_n" in k:
            samples_bias_n[k] = v
            keys_to_remove.append(k)
            
    for k in keys_to_remove:
        del samples_n[k]

    # Rename bias_n samples: b_{i}_slope_n -> b_{angle}_{count}_n
    samples_bias_n_renamed = {}
    
    # Re-use angle counters
    angle_counters_n = {a: 0 for a in exp_angles} # Reset counters
    
    # Sort keys to ensure deterministic ordering (b_1, b_2...)
    # Assuming standard sorting works for b_1, b_2... (needs natsort ideally but simple sort is okay if i<10)
    # Actually, we should parse i again to be safe.
    
    # Collect index-key pairs
    bias_n_items = []
    for k, v in samples_bias_n.items():
        try:
            # b_1_slope_n
            prefix = k.split("_slope_n")[0]
            idx_str = prefix.split("_")[1]
            idx = int(idx_str) - 1
            bias_n_items.append((idx, k, v))
        except:
            samples_bias_n_renamed[k] = v
            
    bias_n_items.sort(key=lambda x: x[0])
    
    for idx, k, v in bias_n_items:
        if 0 <= idx < len(exp_angles):
            ang = exp_angles[idx]
            angle_counters_n[ang] += 1
            count = angle_counters_n[ang]
            
            if angle_counts_total[ang] > 1:
                new_key = f"b_{ang}_{count}_n"
            else:
                new_key = f"b_{ang}_n"
            samples_bias_n_renamed[new_key] = v
        else:
            samples_bias_n_renamed[k] = v

    if samples_bias_n_renamed:
        # User requested specific column layout for these too
        bias_n_data_by_angle = {45: [], 90: [], 135: []}
        
        for k, v in samples_bias_n_renamed.items():
             try:
                 parts = k.split("_")
                 # b_45_1_n -> parts[1] is angle
                 ang = int(parts[1])
                 if ang in bias_n_data_by_angle:
                     bias_n_data_by_angle[ang].append((k, v, k))
             except:
                 continue
                 
        plot_bias_column_layout(
            bias_n_data_by_angle,
            save_path=figures_dir / f"posterior_bias_n_columns_{suffix}.png",
            # No prior plotted: same rationale as b_i (hierarchical structure)
        )
        save_stats_csv(samples_bias_n_renamed, f"inference_bias_n_stats_{suffix}.csv")

    if samples_n:
        plot_posterior_distributions(
            samples_n,
            prior_pdf_fn=get_prior_pdf,
            save_path=figures_dir / f"posterior_n_{suffix}.png",
        )
        save_stats_csv(samples_n, f"inference_n_stats_{suffix}.csv")

    if samples_gamma:
        # Group gamma samples
        # Keys are like "gamma_v_45", "gamma_h_90"
        gamma_groups = {"h": {}, "v": {}}
        
        for key, val in samples_gamma.items():
            try:
                # generic parser
                parts = key.split("_")
                # Expected: gamma, direction, angle
                if len(parts) >= 3:
                     direction = parts[1]
                     angle = int(parts[2])
                     
                     if angle in standard_angles and direction in ["v", "h"]:
                         if angle not in gamma_groups[direction]: gamma_groups[direction][angle] = []
                         # Label: usually just one gamma per direction/angle in this model?
                         # Pass key `key` (e.g. "gamma_v_45") as the 3rd element for prior lookup
                         gamma_groups[direction][angle].append(("Posterior", val, key))
            except (IndexError, ValueError):
                continue
                
        plot_distributions_grid_2x3(
            gamma_groups,
            standard_angles,
            save_path=figures_dir / f"posterior_gamma_grid_{suffix}.png",
            prior_pdf_fn=get_prior_pdf,
            prior_samples=prior_samples_all,
            title_prefix="Gamma"
        )
        save_stats_csv(samples_gamma, f"inference_gamma_stats_{suffix}.csv")

    # 5. Prediction Plots
    print("Generating prediction plots...")

    # Setup prediction points
    preds_angle_cfg = config["data"]["prediction_angle"]
    angles_to_predict = (
        preds_angle_cfg if isinstance(preds_angle_cfg, list) else [preds_angle_cfg]
    )

    # Prediction Intervals
    pi_coverage = config["data"].get("prediction_interval", 0.95)
    alpha = (1.0 - pi_coverage) / 2.0
    q_lower = alpha * 100.0
    q_upper = (1.0 - alpha) * 100.0
    interval_label = f"{int(pi_coverage * 100)}% interval"

    num_pred_samples = config["data"].get("prediction_samples", 500)
    print(
        f"Using {pi_coverage * 100:.0f}% prediction intervals ({q_lower:.1f}% - {q_upper:.1f}%) with {num_pred_samples} samples"
    )
    print(f"Predicting for angles: {angles_to_predict}")

    max_load = config["data"].get("max_load", 10.0)
    samples_load = jnp.linspace(0, max_load, 100)
    # test_xy is angle dependent, moved inside loop

    # Prior samples are now generated earlier
    # prior_samples_all exists

    # Extract only what we need for prediction (exclude lambdas if they are not needed by predict_batch,
    # but actualy predict_batch needs everything).
    # We will pass the full dictionary or extracted arrays?
    # posterior_predict signature:
    # (rng_key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, data_exp, data_sim, test_xy, test_theta,
    #  mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, direction='h')

    # Helper to extract params from samples dict
    def extract_params(samples_dict, idxs=None):
        # Determine number of samples from a common key
        if "E_1" in samples_dict:
            n_samples = samples_dict["E_1"].shape[0]
        elif "mu_emulator" in samples_dict:
            n_samples = samples_dict["mu_emulator"].shape[0]
        else:
            # Fallback
            keys = list(samples_dict.keys())
            if keys:
                n_samples = samples_dict[keys[0]].shape[0]
            else:
                n_samples = 0

        if idxs is None:
            idxs = jnp.arange(n_samples)

        def get_batch(key):
            if key in samples_dict:
                return samples_dict[key][idxs]
            else:
                return jnp.zeros(len(idxs))

        # Physical
        E_1 = get_batch("E_1")
        E_2 = get_batch("E_2")
        v_12 = get_batch("v_12")
        v_23 = get_batch("v_23")
        G_12 = get_batch("G_12")
        test_theta = jnp.stack([E_1, E_2, v_12, v_23, G_12], axis=1)  # (N, 5)

        # Hyper
        # Hyper
        # mu_emulator = get_batch("mu_emulator")
        try:
            mu_emulator_v = get_batch("mu_emulator_v")
        except KeyError:
            mu_emulator_v = jnp.zeros_like(E_1) # Should not happen with new results

        try:
            mu_emulator_h = get_batch("mu_emulator_h")
        except KeyError:
            mu_emulator_h = jnp.zeros_like(E_1)
        sigma_emulator = get_batch("sigma_emulator")

        # Check noise model parameters (backward compatibility)
        # Different noise models sample different parameters
        try:
            sigma_measure = get_batch("sigma_measure")
        except KeyError:
            sigma_measure = jnp.zeros_like(mu_emulator)

        try:
            sigma_measure_base = get_batch("sigma_measure_base")
        except KeyError:
            sigma_measure_base = jnp.zeros_like(mu_emulator)

        try:
            sigma_constant = get_batch("sigma_constant")
        except KeyError:
            sigma_constant = jnp.zeros_like(mu_emulator)

        # Length scales
        l_P = get_batch("lambda_P")
        l_alpha = get_batch("lambda_alpha")

        # Check if these keys exist in samples_dict before accessing
        l_E1 = (
            get_batch("lambda_E1")
            if "lambda_E1" in samples_dict
            else jnp.zeros_like(l_P)
        )  # Placeholder if not present
        l_E2 = (
            get_batch("lambda_E2")
            if "lambda_E2" in samples_dict
            else jnp.zeros_like(l_P)
        )
        l_v12 = (
            get_batch("lambda_v12")
            if "lambda_v12" in samples_dict
            else jnp.zeros_like(l_P)
        )
        l_v23 = (
            get_batch("lambda_v23")
            if "lambda_v23" in samples_dict
            else jnp.zeros_like(l_P)
        )
        l_G12 = (
            get_batch("lambda_G12")
            if "lambda_G12" in samples_dict
            else jnp.zeros_like(l_P)
        )

        length_xy = jnp.stack([l_P, l_alpha], axis=1)
        length_theta = jnp.stack([l_E1, l_E2, l_v12, l_v23, l_G12], axis=1)



         # Empirical Model Params
        bias_slope = []
        if "b_1_slope" in samples_dict:
             # Assume we have b_1_slope to b_{num_exp}_slope
             # We need to find how many experiments.
             # We can count keys starting with "b_" and ending "slope"
             n_exp = 0
             while f"b_{n_exp+1}_slope" in samples_dict:
                 n_exp += 1
             
             for i in range(n_exp):
                 bias_slope.append(get_batch(f"b_{i+1}_slope"))

        # Simple Model 
        betas_simple = None
        sigma_simple = None
        
        # Check if we are in model_simple mode (by checking keys or config?)
        # config is global.
        if config["model_type"] == "model_simple":
             # Extract 6 betas
             betas_simple = {}
             for d in ["v", "h"]:
                 for ang in [45, 90, 135]:
                     k = f"beta_{d}_{ang}"
                     betas_simple[k] = get_batch(k)
             
             # Extract sigma
             sigma_simple = get_batch("sigma_measure")
        
        # If empty (prior or not empirical), might return empty list or None
        # But we need consistent tuple size? 
        # Actually bias_slope is specific to empirical.
        # Let's add it to end of tuple.
        
        return (
            test_theta,
            mu_emulator_v,
            mu_emulator_h,
            sigma_emulator,
            length_xy,
            length_theta,
            sigma_measure,
            sigma_measure_base,
            sigma_constant,
            bias_slope, # List of arrays (N,)
            get_batch("gamma_scale_v") if "gamma_scale_v" in samples_dict else jnp.full(len(idxs), 0.01),
            get_batch("gamma_scale_h") if "gamma_scale_h" in samples_dict else jnp.full(len(idxs), 0.01),
            betas_simple,
            sigma_simple,
        )

    # Extract Prior Params
    prior_params_tuple = extract_params(prior_samples_all)

    # --- Posterior Prediction Setup (ONCE) ---
    # Use a subset of samples
    # Check what key works
    if "E_1" in samples:
         total_samples = samples["E_1"].shape[0]
    elif "beta_v_45" in samples:
         total_samples = samples["beta_v_45"].shape[0]
    elif "mu_emulator_v" in samples:
         total_samples = samples["mu_emulator_v"].shape[0]
    elif "mu_emulator" in samples:
         total_samples = samples["mu_emulator"].shape[0]
    else:
         total_samples = samples["gamma_v_45"].shape[0]  # Fallback for empirical model

    num_samples = min(num_pred_samples, total_samples)
    indices = np.random.choice(total_samples, num_samples, replace=False)

    post_params_tuple = extract_params(samples, indices)


    print("Running predictions per angle...")
    
    # Pre-calculate conditioning data ONCE
    # Full data for posterior
    full_exp_xy = jnp.concatenate(input_xy_exp, axis=0) if len(input_xy_exp) > 0 else jnp.empty((0, 2))
    
    # Concatenate data arrays for posterior
    data_exp_v_concat = jnp.concatenate(data_exp_v, axis=0) if len(data_exp_v) > 0 else jnp.empty((0, 3))
    data_exp_h_concat = jnp.concatenate(data_exp_h, axis=0) if len(data_exp_h) > 0 else jnp.empty((0, 3))
    
    conditioning_data = {
        "exp_xy_full": full_exp_xy,
        "input_xy_sim": input_xy_sim,
        "input_theta_sim": input_theta_sim,
        "data_exp_v_full": data_exp_v_concat,
        "data_exp_h_full": data_exp_h_concat,
        "data_sim_v": data_sim_v,
        "data_sim_h": data_sim_h
    }

    def predict_batch(params_tuple, direction, current_test_xy, use_prior=False):
        """
        Compute GP predictions using pre-compiled JIT function.
        """
        (
            test_theta, mu_em_v, mu_em_h, sig_em, len_xy, len_th, 
            sig_meas, sig_meas_base, sig_const, bias_slope_list, 
            gamma_scale_v_list, gamma_scale_h_list, 
            betas_simple_dict, sigma_simple_list
        ) = params_tuple

        num_samples = test_theta.shape[0]
        rng_keys = random.split(random.PRNGKey(42), num_samples)
        
        # Prepare Conditioning Data
        if use_prior:
            exp_xy = jnp.empty((0, 2))
            exp_data = jnp.empty((0, 3)) 
        else:
            exp_xy = conditioning_data["exp_xy_full"]
            exp_data = conditioning_data["data_exp_v_full"] if direction == "v" else conditioning_data["data_exp_h_full"]
            
        sim_data = conditioning_data["data_sim_v"] if direction == "v" else conditioning_data["data_sim_h"]
        
        # Call JIT function
        means, stds_em, f_samples = predict_fn(
            rng_keys,
            mu_em_h if direction == "h" else mu_em_v,
            sig_em,
            len_xy,
            len_th,
            sig_meas,
            sig_meas_base,
            sig_const,
            bias_slope_list,
            test_theta,
            current_test_xy,
            exp_xy,
            conditioning_data["input_xy_sim"],
            conditioning_data["input_theta_sim"],
            exp_data,
            sim_data,
            direction,
            use_prior,
            gamma_scale_v_list,
            gamma_scale_h_list,
            betas_simple_dict,
            sigma_simple_list,
            config["data"].get("noise_model", "proportional")
        )

        loads = current_test_xy[:, 0]
        n_samples, n_points = f_samples.shape
        noise_model = config["data"].get("noise_model", "proportional")

        if sigma_simple_list is not None:
             noise_std = sigma_simple_list[:, None] * jnp.sqrt(jnp.abs(loads[None, :]) + 1e-6)
             noise_var = noise_std**2
        elif noise_model == "additive":
            noise_var = (sig_meas[:, None] ** 2 * loads[None, :] + sig_meas_base[:, None] ** 2)
        elif noise_model == "constant":
            noise_var = sig_const[:, None] ** 2 * jnp.ones((n_samples, n_points))
        else: # proportional
            noise_var = sig_meas[:, None] ** 2 * loads[None, :]

        total_std = jnp.sqrt(stds_em**2 + noise_var)

        # Observation Samples
        rng_noise = random.PRNGKey(123)
        if sigma_simple_list is not None:
             noise_std = sigma_simple_list[:, None] * jnp.sqrt(jnp.abs(loads[None, :]) + 1e-6)
        else:
             noise_std = jnp.sqrt(noise_var)
        
        noise_samples = noise_std * random.normal(rng_noise, (n_samples, n_points))
        y_samples = f_samples + noise_samples

        return means, total_std, f_samples, y_samples

    predictions_collection = {}

    for angle_value in angles_to_predict:
        predictions_collection[angle_value] = {}
        print(f"\n=== Predicting for Angle {angle_value} ===")

        test_xy = jnp.stack(
            [
                jnp.array([load_val, jnp.deg2rad(angle_value)])
                for load_val in samples_load
            ]
        )

        # Filter pre-loaded data for this angle (Avoid re-loading from disk)
        def filter_by_angle(data_list_xy, data_list_val, target_angle):
            filtered_xy = []
            filtered_val = []
            for xy, val in zip(data_list_xy, data_list_val):
                angle_deg = np.rad2deg(xy[0, 1])
                if np.isclose(angle_deg, target_angle, atol=1.0):
                     filtered_xy.append(xy)
                     filtered_val.append(val)
            return filtered_xy, filtered_val

        filt_xy, filt_h = filter_by_angle(data_dict["input_xy_exp_full"], data_dict["data_exp_h_full_raw"], angle_value)
        _, filt_v = filter_by_angle(data_dict["input_xy_exp_full"], data_dict["data_exp_v_full_raw"], angle_value)
        
        data_current_angle = {
             "input_xy_exp": filt_xy,
             "data_exp_h_raw": filt_h,
             "data_exp_v_raw": filt_v
        }

        for direction in ["v", "h"]:
            dir_label = "Normal" if direction == "v" else "Shear"
            dir_file_tag = "normal" if direction == "v" else "shear"
            
            # Determine Training Status Label
            training_info = None
            model_type = config.get("model_type", "unknown")
            training_direction = config["data"].get("direction", "h") # Default h if missing
            
            if model_type == "model_n":
                if direction != training_direction:
                    training_info = "No training data for this direction"
            
            print(f"  --- {dir_label} Direction ---")

            # Data for plotting
            data_exp_plt = data_current_angle[f"data_exp_{direction}_raw"]
            input_xy_exp_plt = data_current_angle["input_xy_exp"]

            # 1. Prior Prediction
            import time
            t0 = time.time()
            print(f"    Running prior prediction... (Start)")
            # We already have prior samples. Just propagate.
            _, _, prior_f_samples, prior_y_samples = predict_batch(
                prior_params_tuple, direction, test_xy, use_prior=True
            )
            print(f"    Prior prediction done in {time.time()-t0:.2f}s")

            mean_prior = jnp.mean(prior_f_samples, axis=0)  # (100,)
            # Function uncertainty (epistemic only)
            pct_prior_f = jnp.percentile(
                prior_f_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )
            # Observation uncertainty (includes noise)
            pct_prior_y = jnp.percentile(
                prior_y_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )
            
            # --- Verification: Prior Spaghetti - Disabled individual plot ---
            # plot_spaghetti_verification(...)


            # 2. Posterior Prediction
            t0 = time.time()
            print(f"    Running posterior prediction... (Start)")
            _, _, post_f_samples, post_y_samples = predict_batch(
                post_params_tuple, direction, test_xy, use_prior=False
            )
            print(f"    Posterior prediction done in {time.time()-t0:.2f}s")

            mean_post = jnp.mean(post_f_samples, axis=0)
            # Function uncertainty (epistemic only)
            pct_post_f = jnp.percentile(
                post_f_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )
            # Observation uncertainty (includes noise)
            pct_post_y = jnp.percentile(
                post_y_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )

            # 3. Plots - Output Standardization: Disabled individual plots
            # plot_prediction(...) 
            # plot_combined_prediction(...)
            # plot_spaghetti_verification(...)


            # Store for Grid Plot - now with both uncertainty types
            predictions_collection[angle_value][direction] = {
                'samples_load': samples_load,
                'mean_post': mean_post,
                'pct_post_f': pct_post_f,  # Function uncertainty
                'pct_post_y': pct_post_y,  # Observation uncertainty
                'mean_prior': mean_prior,
                'pct_prior_f': pct_prior_f,  # Function uncertainty
                'pct_prior_y': pct_prior_y,  # Observation uncertainty
                'prior_f_samples': prior_f_samples,
                'prior_y_samples': prior_y_samples,
                'input_xy_exp': input_xy_exp_plt,
                'data_exp': data_exp_plt,
                'training_info': training_info,
                'post_f_samples': post_f_samples,
                'post_y_samples': post_y_samples
            }

    # 6. Grid Predictions (Consolidated Spaghetti)
    
    # Prior Grid
    print("\nGenerating Prior Spaghetti Grid...")
    plot_grid_spaghetti(
        predictions_collection,
        angles_to_predict,
        save_path=figures_dir / f"prediction_prior_grid_{suffix}.png",
        title_prefix="Prior"
    )

    # Posterior Grid
    print("Generating Posterior Spaghetti Grid...")
    plot_grid_spaghetti(
        predictions_collection,
        angles_to_predict,
        save_path=figures_dir / f"prediction_posterior_grid_{suffix}.png",
        title_prefix="Posterior"
    )
    # 9. Residual Analysis (Optional)
    if config["data"].get("run_residual_analysis", False) and config.get("model_type") not in ["model_empirical", "model_simple"]:
        run_residual_analysis(idata, data_dict, figures_dir)

    # 10. Traceplots (Optional)
    if config["data"].get("plot_trace", True):
        plot_trace_diagnostics(idata, figures_dir)

    print("Analysis complete.")


def run_residual_analysis(idata, data_dict, figures_dir):
    print("\nRunning Residual Analysis...")

    # Extract data from dictionary
    input_xy_exp = data_dict["input_xy_exp"]
    input_xy_sim = data_dict["input_xy_sim"]
    input_theta_sim = data_dict["input_theta_sim"]
    data_exp_h = data_dict["data_exp_h"]
    data_exp_v = data_dict["data_exp_v"]
    data_sim_h = data_dict["data_sim_h"]
    data_sim_v = data_dict["data_sim_v"]

    # Extract Samples
    num_samples = 500
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    total_samples = posterior.dims.get(
        "sample", posterior.sizes["sample"]
    )  # Handle deprecation safely
    num_samples = min(500, total_samples)
    indices = np.random.choice(total_samples, num_samples, replace=False)

    def get_batch(key):
        val = posterior[key].values
        return jnp.array(val[indices])

    # Physical
    E_1 = get_batch("E_1")
    E_2 = get_batch("E_2")
    v_12 = get_batch("v_12")
    v_23 = get_batch("v_23")
    G_12 = get_batch("G_12")
    test_theta = jnp.stack([E_1, E_2, v_12, v_23, G_12], axis=1)

    # Hyper
    # mu_emulator = get_batch("mu_emulator")
    try: 
        mu_emulator_v = get_batch("mu_emulator_v")
    except KeyError:
        mu_emulator_v = jnp.zeros_like(E_1)

    try:
        mu_emulator_h = get_batch("mu_emulator_h")
    except KeyError:
        mu_emulator_h = jnp.zeros_like(E_1)
    sigma_emulator = get_batch("sigma_emulator")

    try:
        sigma_measure = get_batch("sigma_measure")
    except KeyError:
        sigma_measure = jnp.zeros_like(mu_emulator_v)

    try:
        sigma_measure_base = get_batch("sigma_measure_base")
    except KeyError:
        sigma_measure_base = jnp.zeros_like(mu_emulator_v)

    try:
        sigma_constant = get_batch("sigma_constant")
    except KeyError:
        sigma_constant = jnp.zeros_like(mu_emulator_v)

    # Simple Model 
    betas_simple = None
    sigma_simple = None
    
    # Check if we are in model_simple mode (by checking keys or config?)
    # config is global.
    if config["model_type"] == "model_simple":
         # Extract 6 betas
         betas_simple = {}
         for d in ["v", "h"]:
             for ang in [45, 90, 135]:
                 k = f"beta_{d}_{ang}"
                 betas_simple[k] = get_batch(k)
         
         # Extract sigma
         sigma_simple = get_batch("sigma_epsilon")

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
    def predict_point(
        m_em, s_em, l_xy, l_th, s_meas, s_meas_base, s_const, t_theta, t_xy, direction
    ):
        full_input_xy_exp = jnp.concatenate(
            input_xy_exp, axis=0
        )  # (Total_Exp_Points, 2)
        full_input_theta_exp = jnp.tile(t_theta, (full_input_xy_exp.shape[0], 1))

        full_data_exp = (
            jnp.concatenate(data_exp_v, axis=0)
            if direction == "v"
            else jnp.concatenate(data_exp_h, axis=0)
        )
        trg_data_sim = data_sim_v if direction == "v" else data_sim_h

        mean, std_em, _ = posterior_predict(
            random.PRNGKey(0),
            full_input_xy_exp,
            input_xy_sim,
            full_input_theta_exp,
            input_theta_sim,
            full_data_exp,
            trg_data_sim,
            t_xy,
            t_theta,
            m_em,
            s_em,
            l_xy,
            l_th,
            s_meas,
            s_meas_base,
            s_const,
            direction=direction,
        )
        return mean, std_em

    v_predict = vmap(
        predict_point, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None)
    )

    rows = []
    bands_data = []

    for i in range(len(input_xy_exp)):
        xy_exp = input_xy_exp[i]
        
        # Determine direction
        # Legacy/Simple logic: usually just one direction per experiment?
        # Ideally we check data_exp_h[i] or v[i].
        # For residuals, let's just do both and see which has data?
        # Start with 'v' if data_v[i] has points
        
        directions_to_check = []
        if data_exp_v[i].shape[0] > 0: directions_to_check.append("v")
        if data_exp_h[i].shape[0] > 0: directions_to_check.append("h")
        
        angle_rad = xy_exp[0, 1]
        angle_deg = int(round(np.rad2deg(angle_rad)))
        print(f"  Residuals: Angle {angle_deg}...")

        loads = xy_exp[:, 0]
        max_load = jnp.max(loads)

        for direction in directions_to_check:
            label = "Normal" if direction == "v" else "Shear"
            
            mu_em = mu_emulator_h if direction == "h" else mu_emulator_v
             
            # Predict at observed
            means, stds_em = v_predict(
                mu_em,
                sigma_emulator,
                length_xy,
                length_theta,
                sigma_measure,
                sigma_measure_base,
                sigma_constant,
                test_theta,
                xy_exp,
                direction,
            )

            mu_post = jnp.mean(means, axis=0)

            # Noise model check
            noise_model = config["data"].get("noise_model", "proportional")
            if noise_model == "additive":
                noise_var_samples = (
                    sigma_measure[:, None] ** 2 * loads[None, :]
                    + sigma_measure_base[:, None] ** 2
                )
            elif noise_model == "constant":
                noise_var_samples = sigma_constant[:, None] ** 2 * jnp.ones_like(
                    loads[None, :]
                )
            else:
                noise_var_samples = sigma_measure[:, None] ** 2 * loads[None, :]

            gp_var_samples = stds_em**2
            total_std = jnp.sqrt(
                jnp.mean(gp_var_samples + noise_var_samples, axis=0)
                + jnp.var(means, axis=0)
            )

            # Retrieve raw data for plotting (N, 3)
            # We need to correctly handle if data_exp_raw was passed or if we need to lookup
            # data_exp_h/v were passed as arguments to this function 'generate_prediction_plots'.
            # Inspecting main(), we passed data_dict["data_exp_h_raw"]. So 'data_exp_h' arg IS the raw list.
            
            # Retrieve raw data (N, 3) or (N,)
            # Try to find raw keys in data_dict if possible, but here we only have local vars data_exp_h/v.
            # We should have extracted raw data earlier if we wanted it.
            # Let's assume data_exp_h/v passed to this function might be raw if we changed extraction?
            # No, we extracted data_dict["data_exp_h"] at the top of this function.
            # Let's check keys in data_dict again? data_dict is passed in.
            
            raw_data = None
            raw_key = f"data_exp_{direction}_raw"
            if raw_key in data_dict:
                 raw_data = data_dict[raw_key][i]
            
            if raw_data is None:
                 # Fallback to the extracted data (likely averaged)
                 raw_data = (data_exp_h[i] if direction == "h" else data_exp_v[i])
            
            y_obs_flat = raw_data.flatten()
            
            # Adjust predicted means to match data shape
            if raw_data.ndim > 1 and raw_data.shape[1] == 3:
                # Data is (N, 3) -> repeat means 3 times
                loads_flat = np.repeat(loads, 3)
                mu_post_flat = np.repeat(mu_post, 3)
                total_std_flat = np.repeat(total_std, 3)
            else:
                # Data is (N,) or (N,1) -> 1-to-1
                loads_flat = loads
                mu_post_flat = mu_post
                total_std_flat = total_std
            
            # Residuals
            resid_flat = y_obs_flat - mu_post_flat
            
            # Std for normalization (scalar per load step, repeated)
            total_std_flat = np.repeat(total_std, 3)
            std_resid_flat = resid_flat / total_std_flat

            for k in range(len(loads_flat)):
                rows.append(
                    {
                        "Angle": angle_deg,
                        "Direction": label,
                        "Load": float(loads_flat[k]),
                        "Residual": float(resid_flat[k]),
                        "StdResidual": float(std_resid_flat[k]),
                    }
                )

            # Dense Grid for Smooth Bands
            dense_loads = jnp.linspace(0, max_load, 100)
            dense_angles = jnp.ones_like(dense_loads) * angle_rad
            dense_xy = jnp.stack([dense_loads, dense_angles], axis=1)

            d_means, d_stds_em = v_predict(
                mu_em,
                sigma_emulator,
                length_xy,
                length_theta,
                sigma_measure,
                sigma_measure_base,
                sigma_constant,
                test_theta,
                dense_xy,
                direction,
            )

            if noise_model == "additive":
                d_noise_var = (
                    sigma_measure[:, None] ** 2 * dense_loads[None, :]
                    + sigma_measure_base[:, None] ** 2
                )
            elif noise_model == "constant":
                d_noise_var = sigma_constant[:, None] ** 2 * jnp.ones_like(
                    dense_loads[None, :]
                )
            else:
                d_noise_var = sigma_measure[:, None] ** 2 * dense_loads[None, :]

            d_gp_var = d_stds_em**2
            d_total_std = jnp.sqrt(
                jnp.mean(d_gp_var + d_noise_var, axis=0) + jnp.var(d_means, axis=0)
            )

            bands_data.append(
                {
                    "Angle": angle_deg,
                    "Direction": label,
                    "Load": dense_loads,
                    "SigmaTotal": d_total_std,
                }
            )

    df = pd.DataFrame(rows)

    unique_combos = df[["Angle", "Direction"]].drop_duplicates().values

    # 1. Residuals with Bands
    plt.figure(figsize=(15, 5 * len(unique_combos) // 3 + 5))
    num_plots = len(unique_combos)
    cols = 3
    rows_plt = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_plt, cols, figsize=(15, 4 * rows_plt), constrained_layout=True
    )
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, (angle, direction) in enumerate(unique_combos):
        ax = axes[idx]
        subset = df[(df["Angle"] == angle) & (df["Direction"] == direction)]
        ax.scatter(subset["Load"], subset["Residual"], alpha=0.6, label="Residuals")

        band = next(
            b for b in bands_data if b["Angle"] == angle and b["Direction"] == direction
        )
        upper = 1.96 * band["SigmaTotal"]
        lower = -1.96 * band["SigmaTotal"]

        ax.fill_between(
            band["Load"], lower, upper, color="gray", alpha=0.2, label="95% CI"
        )
        ax.plot(band["Load"], upper, color="gray", linestyle="--", alpha=0.5)
        ax.plot(band["Load"], lower, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="k", linestyle="-", alpha=0.3)
        ax.set_title(f"Angle {angle}° - {direction}")
        ax.set_xlabel("Load [kN]")
        ax.set_ylabel("Residual [mm]")
        if idx == 0:
            ax.legend()

    for i in range(num_plots, len(axes)):
        axes[i].axis("off")
    plt.savefig(figures_dir / "residuals_with_bands.png")

    # 2. Std Residuals vs Load
    g = sns.FacetGrid(
        df,
        col="Angle",
        row="Direction",
        sharex=False,
        sharey=True,
        height=4,
        aspect=1.2,
    )
    g.map(plt.scatter, "Load", "StdResidual", alpha=0.6)
    for ax in g.axes.flat:
        ax.axhline(0, color="k", ls="-", lw=1)
        ax.axhline(2, color="r", ls="--", lw=1)
        ax.axhline(-2, color="r", ls="--", lw=1)
    g.set_axis_labels("Load [kN]", "Standardized Residual")
    g.savefig(figures_dir / "std_residuals_vs_load.png")

    # 3. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df["StdResidual"], kde=True, stat="density", label="Observed")
    x_range = np.linspace(df["StdResidual"].min() - 1, df["StdResidual"].max() + 1, 100)
    from scipy.stats import norm

    plt.plot(x_range, norm.pdf(x_range, 0, 1), "r--", lw=2, label="Standard Normal")
    plt.title("Distribution of Standardized Residuals")
    plt.xlabel("Standardized Residual")
    plt.legend()
    plt.savefig(figures_dir / "std_residuals_hist.png")

    # 4. Q-Q Plot
    plt.figure(figsize=(8, 8))
    import scipy.stats as stats

    stats.probplot(df["StdResidual"], dist="norm", plot=plt)
    plt.title("Q-Q Plot of Standardized Residuals")
    plt.savefig(figures_dir / "residuals_qq.png")

    print("Residual analysis complete.")


def plot_trace_diagnostics(idata, figures_dir):
    """
    Generate and save traceplots for physical parameters and hyperparameters.
    """
    print("\nGenerating Traceplots for Convergence Diagnostics...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group variables
    var_names = list(idata.posterior.data_vars)

    # Identify Physical Parameters (Theta)
    # They usually don't have underscores or are specific names like E_1, v_12, etc.
    # Exclude those ending in '_n' (normalized) or starting with 'mu_'/'sigma_'/'lambda_' unless physical

    # Based on models.py: E_1, E_2, v_12, v_23, G_12 are deterministic or sampled
    # We want the transformed (actual) values if available.

    # Physical params of interest
    physical_params = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    # Bias params
    bias_params = [v for v in var_names if v.startswith("b_") and not v.endswith("_n")]

    # Filter for what's actually in posterior
    physical_vars = [v for v in physical_params if v in var_names]
    bias_vars = [v for v in bias_params if v in var_names]

    # Hyperparameters
    # typically start with mu_, sigma_, lambda_
    hyper_prefixes = ("mu_", "sigma_", "lambda_")
    hyper_vars = [
        v
        for v in var_names
        if v.startswith(hyper_prefixes)
        and not v.endswith("_n")
        and v not in physical_vars
    ]

    # Plot Physical
    if physical_vars:
        print(f"  Plotting trace for physical parameters: {physical_vars}")
        az.plot_trace(idata, var_names=physical_vars)
        plt.tight_layout()
        plt.savefig(figures_dir / f"trace_physical_{timestamp}.png")
        plt.close()

    # Plot Bias
    if bias_vars:
        print(f"  Plotting trace for bias parameters: {len(bias_vars)} variables")
        # Might be too many for one plot if we have many experiments
        # Plot only first few or aggregate if needed. For now, plot all but handle size?
        # If too many, maybe just skip or plot first 5
        if len(bias_vars) > 10:
            print("  Too many bias variables, plotting first 10...")
            bias_vars = bias_vars[:10]

        az.plot_trace(idata, var_names=bias_vars)
        plt.tight_layout()
        plt.savefig(figures_dir / f"trace_bias_{timestamp}.png")
        plt.close()

    # Plot Hyperparameters
    if hyper_vars:
        print(f"  Plotting trace for hyperparameters: {hyper_vars}")
        az.plot_trace(idata, var_names=hyper_vars)
        plt.tight_layout()
        plt.savefig(figures_dir / f"trace_hyper_{timestamp}.png")
        plt.close()

    print("Traceplots saved.")


if __name__ == "__main__":
    main()

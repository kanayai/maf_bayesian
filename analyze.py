
import arviz as az
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from numpyro.infer import Predictive
import pandas as pd
import seaborn as sns
from datetime import datetime


from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, posterior_predict
from src.vis.plotting import plot_experimental_data, plot_posterior_distributions, plot_prediction, plot_combined_prediction

def main():
    print("Starting analysis...")
    
    # 1. Load Data
    data_dict = load_all_data(config)
    
    # 2. Plot Experimental Data
    # Setup output directory
    model_type = config.get("model_type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figures_dir = Path("figures") / f"analysis_{model_type}_{timestamp}"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {figures_dir}")
    
    plot_experimental_data(data_dict, save_path=figures_dir / "experimental_data.png")
    
    # 3. Load Latest Results
    results_dir = Path("results")
    files = list(results_dir.glob("*.nc"))
    if not files:
        print("No result files found in results/")
        return
        
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"Loading results from {latest_file}")
    idata = az.from_netcdf(latest_file)
    
    # 4. Categorized Posterior Plots
    samples = {}
    for key in idata.posterior.data_vars:
        data = idata.posterior[key].values
        flat_data = data.reshape(-1, *data.shape[2:])
        samples[key] = jnp.array(flat_data)
        
    # Define categories
    physical_params = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    hyper_params = ["mu_emulator", "sigma_emulator", "sigma_measure", 
                    "lambda_P", "lambda_alpha", "lambda_E1", "lambda_E2", 
                    "lambda_v12", "lambda_v23", "lambda_G12"]
    
    # Filter samples into categories
    samples_physical = {k: v for k, v in samples.items() if k in physical_params}
    samples_hyper = {k: v for k, v in samples.items() if k in hyper_params}
    samples_bias = {k: v for k, v in samples.items() if k.startswith("b_") or k.startswith("sigma_b")}
    # Separate _n parameters (normalized)
    samples_n = {k: v for k, v in samples.items() if k.endswith("_n")}
    
    # Unpack data
    input_xy_exp = data_dict["input_xy_exp"]
    input_xy_sim = data_dict["input_xy_sim"]
    input_theta_sim = data_dict["input_theta_sim"]
    data_exp_h = data_dict["data_exp_h"]
    data_exp_v = data_dict["data_exp_v"]
    data_sim_h = data_dict["data_sim_h"]
    data_sim_v = data_dict["data_sim_v"]
    
    # --- Analytical Priors ---
    # We construct functions that return the log_prob (or pdf) for plotting
    # For reparameterized priors (model_n_hv), the physical params E_1 etc are:
    # val = mean + scale * N(0,1) -> val ~ N(mean, scale)
    
    from scipy.stats import norm, expon, lognorm
    
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

        # Hyperparameters (Lambdas)
        # In model_n_hv: lambda_P is sampled as Normal(0, 1).
        # So if we plot lambda_P, the prior is N(0, 1).
        if key.startswith("lambda_"):
            # lambda ~ LogNormal(mean, scale)
            # In scipy.stats.lognorm(s, scale):
            # s = sigma (scale in config)
            # scale = exp(mu) (exp(mean) in config)
            if key in priors_config["hyper"]["length_scales"]:
                p = priors_config["hyper"]["length_scales"][key]
                s = p["scale"]
                scale = np.exp(p["mean"])
                return lognorm.pdf(x_vals, s=s, scale=scale)
            return None
            
        # Emulator Mean/Scale
        # mu_emulator: Normal(mean, scale)
        if key == "mu_emulator":
            p = priors_config["hyper"]["mu_emulator"]
            return norm.pdf(x_vals, loc=p["mean"], scale=p["scale"])
            
        # sigma_emulator: Exponential(20) -> This is target dist for reparam?
        # In model: sigma_emulator_n ~ N(0,1)
        # sigma_emulator = TransformedDistribution(Normal(0,1), InverseCDFTransform(Exponential(20)))
        # Wait, let's check models.py to be sure about sigma_emulator.
        # It uses get_priors_from_config which does:
        # base = sample(name + "_n", Normal(0,1))
        # val = Deterministic(name, target_dist.icdf(Normal(0,1).cdf(base)))
        # So 'val' follows 'target_dist'.
        if key == "sigma_emulator":
             # Exponential(rate) -> pdf = rate * exp(-rate * x)
             # config has "target_dist": dist.Exponential(20.)
             # We need the rate. 
             # dist.Exponential(rate)
             rate = 20.0
             return expon.pdf(x_vals, scale=1/rate) # scipy uses scale=1/rate
             
        if key == "sigma_measure":
             rate = 100.0
             return expon.pdf(x_vals, scale=1/rate)

        # Bias
        if key.startswith("b_"):
            # b_E1 ~ Normal(0, sigma_b_E1)
            # This is hierarchical. We can't easily plot a marginal prior for b_E1 without integrating out sigma_b_E1.
            # Or maybe sigma_b_E1 is fixed? No, it has a prior.
            # For now, maybe skip prior for bias terms or just plot N(0, 1) if they were normalized?
            # They are: b_E1 = sigma_b_E1 * b_E1_n. b_E1_n ~ N(0,1).
            return None
            
        if key.startswith("sigma_b"):
            # Exponential priors
            if "E1" in key: return expon.pdf(x_vals, scale=1/0.0001)
            if "alpha" in key: return expon.pdf(x_vals, scale=np.deg2rad(10)) # 1/rate = scale. rate=1/val -> scale=val
            
        # Normalized params (_n)
        if key.startswith("lambda_"):
            if key in priors_config["hyper"]["length_scales"]:
                p = priors_config["hyper"]["length_scales"][key]
                s = p["scale"]
                scale = np.exp(p["mean"])
                return lognorm.pdf(x_vals, s=s, scale=scale)
            return norm.pdf(x_vals, loc=0.0, scale=1.0)
            
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
            
            stats.append({
                "Parameter": key,
                "Mean": mean,
                "Variance": var,
                "Std": std
            })
        
        df = pd.DataFrame(stats)
        df.set_index("Parameter", inplace=True)
        df.to_csv(figures_dir / filename)
        print(f"Saved stats to {figures_dir / filename}")

    # Filename suffix
    suffix = timestamp

    # Enforce order for physical params
    # E_1, E_2, v_12, v_23, G_12
    ordered_physical = ["E_1", "E_2", "v_12", "v_23", "G_12"]
    samples_physical_ordered = {k: samples_physical[k] for k in ordered_physical if k in samples_physical}

    if samples_physical_ordered:
        plot_posterior_distributions(samples_physical_ordered, prior_pdf_fn=get_prior_pdf, 
                                     save_path=figures_dir / f"posterior_physical_{suffix}.png",
                                     layout_rows=1) # Force 1 row
        save_stats_csv(samples_physical_ordered, f"inference_theta_stats_{suffix}.csv")
        
    if samples_hyper:
        plot_posterior_distributions(samples_hyper, prior_pdf_fn=get_prior_pdf, 
                                     save_path=figures_dir / f"posterior_hyper_{suffix}.png")
        save_stats_csv(samples_hyper, f"inference_hyper_stats_{suffix}.csv")
        
    if samples_bias:
        plot_posterior_distributions(samples_bias, prior_pdf_fn=get_prior_pdf, 
                                     save_path=figures_dir / f"posterior_bias_{suffix}.png")
        save_stats_csv(samples_bias, f"inference_bias_stats_{suffix}.csv")

    if samples_n:
        plot_posterior_distributions(samples_n, prior_pdf_fn=get_prior_pdf, 
                                     save_path=figures_dir / f"posterior_n_{suffix}.png")
        save_stats_csv(samples_n, f"inference_n_stats_{suffix}.csv")
        
    # 5. Prediction Plots
    print("Generating prediction plots...")
    
    # Setup prediction points
    preds_angle_cfg = config["data"]["prediction_angle"]
    angles_to_predict = preds_angle_cfg if isinstance(preds_angle_cfg, list) else [preds_angle_cfg]
    
    # Prediction Intervals
    pi_coverage = config["data"].get("prediction_interval", 0.95)
    alpha = (1.0 - pi_coverage) / 2.0
    q_lower = alpha * 100.0
    q_upper = (1.0 - alpha) * 100.0
    interval_label = f"{int(pi_coverage*100)}% interval"
    
    num_pred_samples = config["data"].get("prediction_samples", 500)
    print(f"Using {pi_coverage*100:.0f}% prediction intervals ({q_lower:.1f}% - {q_upper:.1f}%) with {num_pred_samples} samples")
    print(f"Predicting for angles: {angles_to_predict}")
    
    samples_load = jnp.linspace(0, 10, 100)
    # test_xy is angle dependent, moved inside loop
    
    # --- Generate Prior Samples for Prediction (ONCE) ---
    print("Generating prior samples for prediction...")
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)
    
    # We need to run Predictive to get priors for all variables
    prior_predictive = Predictive(model_n_hv, num_samples=num_pred_samples)
    prior_samples_all = prior_predictive(rng_key_prior, input_xy_exp, input_xy_sim, input_theta_sim, 
                                     data_exp_h, data_exp_v, data_sim_h, data_sim_v, config)
    
    # Extract only what we need for prediction (exclude lambdas if they are not needed by predict_batch, 
    # but actualy predict_batch needs everything).
    # We will pass the full dictionary or extracted arrays?
    # posterior_predict signature:
    # (rng_key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, data_exp, data_sim, test_xy, test_theta, 
    #  mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, direction='h')
    
    # Helper to extract params from samples dict
    def extract_params(samples_dict, idxs=None):
        if idxs is None: idxs = jnp.arange(samples_dict["E_1"].shape[0])
        
        # Physical
        E_1 = samples_dict["E_1"][idxs]
        E_2 = samples_dict["E_2"][idxs]
        v_12 = samples_dict["v_12"][idxs]
        v_23 = samples_dict["v_23"][idxs]
        G_12 = samples_dict["G_12"][idxs]
        test_theta = jnp.stack([E_1, E_2, v_12, v_23, G_12], axis=1) # (N, 5)
        
        # Hyper
        mu_emulator = samples_dict["mu_emulator"][idxs]
        sigma_emulator = samples_dict["sigma_emulator"][idxs]
        sigma_measure = samples_dict["sigma_measure"][idxs]
        
        # Length scales
        l_P = samples_dict["lambda_P"][idxs]
        l_alpha = samples_dict["lambda_alpha"][idxs]
    
        # Check if these keys exist in samples_dict before accessing
        l_E1 = samples_dict["lambda_E1"][idxs] if "lambda_E1" in samples_dict else jnp.zeros_like(l_P) # Placeholder if not present
        l_E2 = samples_dict["lambda_E2"][idxs] if "lambda_E2" in samples_dict else jnp.zeros_like(l_P)
        l_v12 = samples_dict["lambda_v12"][idxs] if "lambda_v12" in samples_dict else jnp.zeros_like(l_P)
        l_v23 = samples_dict["lambda_v23"][idxs] if "lambda_v23" in samples_dict else jnp.zeros_like(l_P)
        l_G12 = samples_dict["lambda_G12"][idxs] if "lambda_G12" in samples_dict else jnp.zeros_like(l_P)
        
        length_xy = jnp.stack([l_P, l_alpha], axis=1)
        length_theta = jnp.stack([l_E1, l_E2, l_v12, l_v23, l_G12], axis=1)
        
        return test_theta, mu_emulator, sigma_emulator, length_xy, length_theta, sigma_measure

    # Extract Prior Params
    prior_params_tuple = extract_params(prior_samples_all)

    # --- Posterior Prediction Setup (ONCE) ---
    # Use a subset of samples
    num_samples = min(num_pred_samples, samples["E_1"].shape[0])
    indices = np.random.choice(samples["E_1"].shape[0], num_samples, replace=False)
    
    post_params_tuple = extract_params(samples, indices)
    
    print("Running predictions per angle...")
    
    # Helper to batched predict
    def predict_batch(params_tuple, direction, current_test_xy):
        test_theta, mu_em, sig_em, len_xy, len_th, sig_meas = params_tuple
        
        # Vectorize posterior_predict over samples
        # posterior_predict(rng_key, ..., test_theta, mean_emulator, ..., direction)
        
        # We fix the experimental/simulation inputs (they are constant/global)
        # We vary the parameters (test_theta, etc.)
        
        def single_pred(rng, t_th, m_em, s_em, l_xy, l_th, s_meas):
            # Tile theta for experimental data size
            input_xy_exp_concat = jnp.concatenate(input_xy_exp, axis=0)
            input_theta_exp_concat = jnp.tile(t_th, (input_xy_exp_concat.shape[0], 1))
            
            # Target data for conditioning based on direction
            data_exp_target = jnp.concatenate(data_exp_v, axis=0) if direction == 'v' else jnp.concatenate(data_exp_h, axis=0)
            data_sim_target = data_sim_v if direction == 'v' else data_sim_h
            
            return posterior_predict(rng, input_xy_exp_concat, input_xy_sim, input_theta_exp_concat, input_theta_sim, 
                                     data_exp_target, data_sim_target, current_test_xy, t_th, 
                                     m_em, s_em, l_xy, l_th, s_meas, direction=direction)
                                   
        # vmap over the 0-th dimension of all mapped args
        batch_pred = vmap(single_pred)(random.split(random.PRNGKey(1), test_theta.shape[0]), 
                                     test_theta, mu_em, sig_em, len_xy, len_th, sig_meas)
                                     
        # Returns: mean_post, stdev_post, sample_post
        return batch_pred

    for angle_value in angles_to_predict:
        print(f"\n=== Predicting for Angle {angle_value} ===")
        
        test_xy = jnp.stack([jnp.array([l, jnp.deg2rad(angle_value)]) for l in samples_load])
        
        # Re-load data for this specific angle to get the points for plotting
        # Create a temp config with only this angle
        import copy
        temp_config = copy.deepcopy(config)
        temp_config["data"]["angles"] = [angle_value]
        data_current_angle = load_all_data(temp_config)
        
        for direction in ['v', 'h']:
            dir_label = "Normal" if direction == 'v' else "Shear"
            dir_file_tag = "normal" if direction == 'v' else "shear"
            print(f"  --- {dir_label} Direction ---")
            
            # Data for plotting
            data_exp_plt = data_current_angle[f"data_exp_{direction}_raw"]
            input_xy_exp_plt = data_current_angle["input_xy_exp"]
            
            # 1. Prior Prediction
            print("    Running prior prediction...")
            # We already have prior samples. Just propagate.
            _, _, prior_y_samples = predict_batch(prior_params_tuple, direction, test_xy)
            
            mean_prior = jnp.mean(prior_y_samples, axis=0) # (100,)
            pct_prior = jnp.percentile(prior_y_samples, q=jnp.array([q_lower, q_upper]), axis=0)
            
            # 2. Posterior Prediction
            print("    Running posterior prediction...")
            _, _, post_y_samples = predict_batch(post_params_tuple, direction, test_xy)
            
            mean_post = jnp.mean(post_y_samples, axis=0)
            pct_post = jnp.percentile(post_y_samples, q=jnp.array([q_lower, q_upper]), axis=0)
            
            # 3. Plots
            print(f"    Generating plots for {dir_label}...")
            
            # Posterior Prediction Plot
            plot_prediction(samples_load, mean_post, pct_post, input_xy_exp_plt, data_exp_plt, angle_value, 
                            f"Posterior Prediction ({dir_label})", 
                            save_path=figures_dir / f"prediction_posterior_{angle_value}_{dir_file_tag}_{suffix}.png",
                            interval_label=interval_label)

            # Prior Prediction Plot
            plot_prediction(samples_load, mean_prior, pct_prior, input_xy_exp_plt, data_exp_plt, angle_value, 
                            f"Prior Prediction ({dir_label})", 
                            save_path=figures_dir / f"prediction_prior_{angle_value}_{dir_file_tag}_{suffix}.png",
                            interval_label=interval_label)
                            
            # Combined Prediction Plot
            plot_combined_prediction(samples_load, mean_prior, pct_prior, mean_post, pct_post, 
                                     input_xy_exp_plt, data_exp_plt, angle_value, 
                                     f"Predictions ({dir_label})",
                                     save_path=figures_dir / f"prediction_combined_{angle_value}_{dir_file_tag}_{suffix}.png",
                                     interval_label=interval_label)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()


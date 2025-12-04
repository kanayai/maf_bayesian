import arviz as az
import jax.numpy as jnp
import jax.random as random
from jax import vmap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from numpyro.infer import Predictive

from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, posterior_predict
from src.vis.plotting import plot_experimental_data, plot_posterior_distributions, plot_prediction

def main():
    print("Starting analysis...")
    
    # 1. Load Data
    data_dict = load_all_data(config)
    
    # 2. Plot Experimental Data
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
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
    
    import jax.scipy.stats.norm as norm
    import jax.scipy.stats.expon as expon
    
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
            return norm.pdf(x_vals, loc=0.0, scale=1.0)
            
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
        if key.endswith("_n"):
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
    from datetime import datetime
    suffix = datetime.now().strftime("%Y%m%d_%H%M")

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
    angle_value = config["data"]["prediction_angle"]
    samples_load = jnp.linspace(0, 10, 100)
    test_xy = jnp.stack([jnp.array([l, jnp.deg2rad(angle_value)]) for l in samples_load])
    
    # --- Generate Prior Samples for Prediction ---
    print("Generating prior samples for prediction...")
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)
    
    # We need to run Predictive to get priors for all variables
    prior_predictive = Predictive(model_n_hv, num_samples=500)
    prior_samples_all = prior_predictive(rng_key_prior, input_xy_exp, input_xy_sim, input_theta_sim, 
                                     data_exp_h, data_exp_v, data_sim_h, data_sim_v, config)
    
    # --- Prior Prediction ---
    print("Running prior prediction...")
    
    # Extract prior params for prediction function
    def extract_params(s):
        return (s["E_1"], s["E_2"], s["v_12"], s["v_23"], s["G_12"],
                s["mu_emulator"], s["sigma_emulator"], s["sigma_measure"],
                s["lambda_P"], s["lambda_alpha"], s["lambda_E1"], s["lambda_E2"],
                s["lambda_v12"], s["lambda_v23"], s["lambda_G12"])

    prior_extracted = extract_params(prior_samples_all)
    
    # Vectorized prediction helper
    def predict_batch(rng_key, theta_vals, hyper_vals):
        (E1, E2, v12, v23, G12, mu, sigma_e, sigma_m, lP, la, lE1, lE2, lv12, lv23, lG12) = theta_vals + hyper_vals
        
        theta = jnp.stack([E1, E2, v12, v23, G12], axis=1)
        length_xy = jnp.stack([lP, la], axis=1)
        length_theta = jnp.stack([lE1, lE2, lv12, lv23, lG12], axis=1)
        
        def single_predict(key, t, me, se, lx, lt, sm):
            # Tile theta for experimental data size (using first dataset as reference size)
            # Note: This assumes all exp datasets share the same theta (no bias or handled outside)
            # In posterior_predict, input_theta_exp is passed.
            # For simplicity, we assume no bias here or that theta includes bias if needed.
            # But posterior_predict takes input_theta_exp.
            # We will tile 't' to match input_xy_exp size.
            
            # We need to handle the list of inputs for input_xy_exp
            # posterior_predict expects concatenated arrays?
            # Let's check posterior_predict signature: 
            # input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim...
            
            # We need to concatenate the experimental inputs for the prediction function
            input_xy_exp_concat = jnp.concatenate(input_xy_exp, axis=0)
            input_theta_exp_concat = jnp.tile(t, (input_xy_exp_concat.shape[0], 1))
            
            # Target data for conditioning
            direction = config["data"]["direction"]
            data_exp_target = jnp.concatenate(data_exp_v, axis=0) if direction == 'v' else jnp.concatenate(data_exp_h, axis=0)
            data_sim_target = data_sim_v if direction == 'v' else data_sim_h
            
            return posterior_predict(key, input_xy_exp_concat, input_xy_sim, input_theta_exp_concat, input_theta_sim, 
                                     data_exp_target, data_sim_target, test_xy, t, me, se, lx, lt, sm, direction=direction)[2]

        return vmap(single_predict)(random.split(rng_key, theta.shape[0]), theta, mu, sigma_e, length_xy, length_theta, sigma_m)

    prior_preds = predict_batch(rng_key_prior, prior_extracted[:5], prior_extracted[5:])
    mean_prior = jnp.mean(prior_preds, axis=0)
    pct_prior = jnp.percentile(prior_preds, jnp.array([5.0, 95.0]), axis=0)
    
    # --- Posterior Prediction ---
    print("Running posterior prediction...")
    # Use a subset of samples
    num_samples = min(500, samples["E_1"].shape[0])
    indices = np.random.choice(samples["E_1"].shape[0], num_samples, replace=False)
    
    post_extracted = [samples[k][indices] for k in ["E_1", "E_2", "v_12", "v_23", "G_12",
                                                    "mu_emulator", "sigma_emulator", "sigma_measure",
                                                    "lambda_P", "lambda_alpha", "lambda_E1", "lambda_E2",
                                                    "lambda_v12", "lambda_v23", "lambda_G12"]]
    
    post_preds = predict_batch(rng_key_post, post_extracted[:5], post_extracted[5:])
    mean_post = jnp.mean(post_preds, axis=0)
    pct_post = jnp.percentile(post_preds, jnp.array([5.0, 95.0]), axis=0)
    
    # --- Plotting ---
    from src.vis.plotting import plot_combined_prediction
    
    # Prepare data for plotting (filtering by angle)
    input_xy_exp_plt = []
    data_exp_plt = []
    direction = config["data"]["direction"]
    data_exp_raw = data_dict["data_exp_v_raw"] if direction == 'v' else data_dict["data_exp_h_raw"]
    
    for i, inp in enumerate(input_xy_exp):
        ang = np.rad2deg(inp[0, 1])
        if np.isclose(ang, angle_value, atol=1e-1):
            input_xy_exp_plt.append(inp)
            data_exp_plt.append(data_exp_raw[i]) # Use raw data (all sensors) for plotting
            
    # 1. Posterior Prediction Plot
    plot_prediction(samples_load, mean_post, pct_post, input_xy_exp_plt, data_exp_plt, angle_value, "Posterior Prediction", 
                    save_path=figures_dir / f"prediction_posterior_{angle_value}_{suffix}.png")

    # 2. Prior Prediction Plot (using same function, labeled as Prior)
    plot_prediction(samples_load, mean_prior, pct_prior, input_xy_exp_plt, data_exp_plt, angle_value, "Prior Prediction", 
                    save_path=figures_dir / f"prediction_prior_{angle_value}_{suffix}.png")
                    
    # 3. Combined Prediction Plot
    plot_combined_prediction(samples_load, mean_prior, pct_prior, mean_post, pct_post, 
                             input_xy_exp_plt, data_exp_plt, angle_value, "Predictions",
                             save_path=figures_dir / f"prediction_combined_{angle_value}_{suffix}.png")
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()

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
import argparse


from configs.default_config import config
from src.io.data_loader import load_all_data
from src.core.models import model_n_hv, posterior_predict
from src.vis.plotting import (
    plot_experimental_data,
    plot_posterior_distributions,
    plot_prediction,
    plot_combined_prediction,
)


def main():
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

    # 2. Plot Experimental Data
    # Setup output directory
    model_type = config.get("model_type", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    hyper_params = [
        "mu_emulator",
        "sigma_emulator",
        "sigma_measure",
        "sigma_measure_base",
        "sigma_constant",
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
        if k.startswith("b_") or k.startswith("sigma_b")
    }
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
            return expon.pdf(x_vals, scale=1 / rate)

        # Hyperparameters - sigma_measure, sigma_emulator, sigma_constant
        if key == "sigma_measure":
            # Exponential(100) dist
            return expon.pdf(x_vals, scale=1 / 100.0)
        if key == "sigma_measure_base":
            return expon.pdf(x_vals, scale=1 / 100.0)
        if key == "sigma_constant":
            # Exponential(0.1) dist -> scale = 1/0.1 = 10
            return expon.pdf(x_vals, scale=10.0)

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
            if "E1" in key:
                return expon.pdf(x_vals, scale=1 / 0.0001)
            if "alpha" in key:
                return expon.pdf(
                    x_vals, scale=np.deg2rad(10)
                )  # 1/rate = scale. rate=1/val -> scale=val

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

            stats.append({"Parameter": key, "Mean": mean, "Variance": var, "Std": std})

        df = pd.DataFrame(stats)
        df.set_index("Parameter", inplace=True)
        df.to_csv(figures_dir / filename)
        print(f"Saved stats to {figures_dir / filename}")

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
        plot_posterior_distributions(
            samples_hyper,
            prior_pdf_fn=get_prior_pdf,
            save_path=figures_dir / f"posterior_hyper_{suffix}.png",
        )
        save_stats_csv(samples_hyper, f"inference_hyper_stats_{suffix}.csv")

    if samples_bias:
        plot_posterior_distributions(
            samples_bias,
            prior_pdf_fn=get_prior_pdf,
            save_path=figures_dir / f"posterior_bias_{suffix}.png",
        )
        save_stats_csv(samples_bias, f"inference_bias_stats_{suffix}.csv")

    if samples_n:
        plot_posterior_distributions(
            samples_n,
            prior_pdf_fn=get_prior_pdf,
            save_path=figures_dir / f"posterior_n_{suffix}.png",
        )
        save_stats_csv(samples_n, f"inference_n_stats_{suffix}.csv")

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

    samples_load = jnp.linspace(0, 10, 100)
    # test_xy is angle dependent, moved inside loop

    # --- Generate Prior Samples for Prediction (ONCE) ---
    print("Generating prior samples for prediction...")
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_prior, rng_key_post = random.split(rng_key, 3)

    # We need to run Predictive to get priors for all variables
    prior_predictive = Predictive(model_n_hv, num_samples=num_pred_samples)
    prior_samples_all = prior_predictive(
        rng_key_prior,
        input_xy_exp,
        input_xy_sim,
        input_theta_sim,
        data_exp_h,
        data_exp_v,
        data_sim_h,
        data_sim_v,
        config,
    )

    # Extract only what we need for prediction (exclude lambdas if they are not needed by predict_batch,
    # but actualy predict_batch needs everything).
    # We will pass the full dictionary or extracted arrays?
    # posterior_predict signature:
    # (rng_key, input_xy_exp, input_xy_sim, input_theta_exp, input_theta_sim, data_exp, data_sim, test_xy, test_theta,
    #  mean_emulator, stdev_emulator, length_xy, length_theta, stdev_measure, direction='h')

    # Helper to extract params from samples dict
    def extract_params(samples_dict, idxs=None):
        if idxs is None:
            idxs = jnp.arange(samples_dict["E_1"].shape[0])

        def get_batch(key):
            return samples_dict[key][idxs]

        # Physical
        E_1 = get_batch("E_1")
        E_2 = get_batch("E_2")
        v_12 = get_batch("v_12")
        v_23 = get_batch("v_23")
        G_12 = get_batch("G_12")
        test_theta = jnp.stack([E_1, E_2, v_12, v_23, G_12], axis=1)  # (N, 5)

        # Hyper
        mu_emulator = get_batch("mu_emulator")
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

        return (
            test_theta,
            mu_emulator,
            sigma_emulator,
            length_xy,
            length_theta,
            sigma_measure,
            sigma_measure_base,
            sigma_constant,
        )

    # Extract Prior Params
    prior_params_tuple = extract_params(prior_samples_all)

    # --- Posterior Prediction Setup (ONCE) ---
    # Use a subset of samples
    num_samples = min(num_pred_samples, samples["E_1"].shape[0])
    indices = np.random.choice(samples["E_1"].shape[0], num_samples, replace=False)

    post_params_tuple = extract_params(samples, indices)

    print("Running predictions per angle...")

    def predict_batch(params_tuple, direction, current_test_xy, use_prior=False):
        """
        Compute GP predictions using posterior_predict.

        Args:
            params_tuple: Tuple of parameter samples
            direction: 'h' or 'v' for horizontal/vertical
            current_test_xy: Test points to predict at
            use_prior: If True, condition only on simulation data (prior predictions).
                      If False, condition on both simulation and experimental data (posterior).
        """
        (
            test_theta,
            mu_em,
            sig_em,
            len_xy,
            len_th,
            sig_meas,
            sig_meas_base,
            sig_const,
        ) = params_tuple

        def predict_point(
            rng,
            m_em,
            s_em,
            l_xy,
            l_th,
            s_meas,
            s_meas_base,
            s_const,
            t_theta,
            t_xy,
            direction,
            is_prior,
        ):
            if is_prior:
                # Prior: condition only on simulation data
                exp_xy = jnp.empty((0, 2))
                exp_theta = jnp.empty((0, 5))
                exp_data = jnp.empty(0)
            else:
                # Posterior: condition on both simulation and experimental data
                exp_xy = jnp.concatenate(input_xy_exp, axis=0)
                exp_theta = jnp.tile(t_theta, (exp_xy.shape[0], 1))
                exp_data = jnp.concatenate(
                    data_exp_v if direction == "v" else data_exp_h, axis=0
                )

            sim_data = data_sim_v if direction == "v" else data_sim_h

            # Get GP realization via Cholesky sampling from posterior_predict
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
            )
            return mean, std_em, sample

        vmap_predict = vmap(
            predict_point, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None)
        )

        num_samples = test_theta.shape[0]
        rng_keys = random.split(random.PRNGKey(42), num_samples)

        means, stds_em, y_samples = vmap_predict(
            rng_keys,
            mu_em,
            sig_em,
            len_xy,
            len_th,
            sig_meas,
            sig_meas_base,
            sig_const,
            test_theta,
            current_test_xy,
            direction,
            use_prior,
        )

        # Compute total_std for reporting
        loads = current_test_xy[:, 0]
        noise_model = config["data"].get("noise_model", "proportional")

        if noise_model == "additive":
            noise_var = (
                sig_meas[:, None] ** 2 * loads[None, :] + sig_meas_base[:, None] ** 2
            )
        elif noise_model == "constant":
            noise_var = sig_const[:, None] ** 2 * jnp.ones_like(loads[None, :])
        else:
            noise_var = sig_meas[:, None] ** 2 * loads[None, :]

        total_std = jnp.sqrt(stds_em**2 + noise_var)

        return means, total_std, y_samples

    for angle_value in angles_to_predict:
        print(f"\n=== Predicting for Angle {angle_value} ===")

        test_xy = jnp.stack(
            [jnp.array([l, jnp.deg2rad(angle_value)]) for l in samples_load]
        )

        # Re-load data for this specific angle to get the points for plotting
        # Create a temp config with only this angle
        import copy

        temp_config = copy.deepcopy(config)
        temp_config["data"]["angles"] = [angle_value]
        data_current_angle = load_all_data(temp_config)

        for direction in ["v", "h"]:
            dir_label = "Normal" if direction == "v" else "Shear"
            dir_file_tag = "normal" if direction == "v" else "shear"
            print(f"  --- {dir_label} Direction ---")

            # Data for plotting
            data_exp_plt = data_current_angle[f"data_exp_{direction}_raw"]
            input_xy_exp_plt = data_current_angle["input_xy_exp"]

            # 1. Prior Prediction
            print("    Running prior prediction...")
            # We already have prior samples. Just propagate.
            _, _, prior_y_samples = predict_batch(
                prior_params_tuple, direction, test_xy, use_prior=True
            )

            mean_prior = jnp.mean(prior_y_samples, axis=0)  # (100,)
            pct_prior = jnp.percentile(
                prior_y_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )

            # 2. Posterior Prediction
            print("    Running posterior prediction...")
            _, _, post_y_samples = predict_batch(
                post_params_tuple, direction, test_xy, use_prior=False
            )

            mean_post = jnp.mean(post_y_samples, axis=0)
            pct_post = jnp.percentile(
                post_y_samples, q=jnp.array([q_lower, q_upper]), axis=0
            )

            # 3. Plots
            print(f"    Generating plots for {dir_label}...")

            # Posterior Prediction Plot
            plot_prediction(
                samples_load,
                mean_post,
                pct_post,
                input_xy_exp_plt,
                data_exp_plt,
                angle_value,
                f"Posterior Prediction ({dir_label})",
                save_path=figures_dir
                / f"prediction_posterior_{angle_value}_{dir_file_tag}_{suffix}.png",
                interval_label=interval_label,
            )

            # Prior Prediction Plot
            plot_prediction(
                samples_load,
                mean_prior,
                pct_prior,
                input_xy_exp_plt,
                data_exp_plt,
                angle_value,
                f"Prior Prediction ({dir_label})",
                save_path=figures_dir
                / f"prediction_prior_{angle_value}_{dir_file_tag}_{suffix}.png",
                interval_label=interval_label,
            )

            # Combined Prediction Plot
            plot_combined_prediction(
                samples_load,
                mean_prior,
                pct_prior,
                mean_post,
                pct_post,
                input_xy_exp_plt,
                data_exp_plt,
                angle_value,
                f"Predictions ({dir_label})",
                save_path=figures_dir
                / f"prediction_combined_{angle_value}_{dir_file_tag}_{suffix}.png",
                interval_label=interval_label,
            )

    # 9. Residual Analysis (Optional)
    if config["data"].get("run_residual_analysis", False):
        run_residual_analysis(idata, data_dict, figures_dir)

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
    mu_emulator = get_batch("mu_emulator")
    sigma_emulator = get_batch("sigma_emulator")
    sigma_measure = get_batch("sigma_measure")
    try:
        sigma_measure_base = get_batch("sigma_measure_base")
    except KeyError:
        sigma_measure_base = jnp.zeros_like(sigma_measure)

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
        m_em, s_em, l_xy, l_th, s_meas, s_meas_base, t_theta, t_xy, direction
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
            direction=direction,
        )
        return mean, std_em

    vmap_predict = vmap(predict_point, in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))

    rows = []
    bands_data = []

    for i, xy_exp in enumerate(input_xy_exp):
        angle_rad = xy_exp[0, 1]
        angle_deg = int(round(np.rad2deg(angle_rad)))
        print(f"  Residuals: Angle {angle_deg}...")

        loads = xy_exp[:, 0]
        max_load = jnp.max(loads)

        for direction, label in [("h", "Shear"), ("v", "Normal")]:
            # Predict at observed
            means, stds_em = vmap_predict(
                mu_emulator,
                sigma_emulator,
                length_xy,
                length_theta,
                sigma_measure,
                sigma_measure_base,
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
            else:
                noise_var_samples = sigma_measure[:, None] ** 2 * loads[None, :]

            gp_var_samples = stds_em**2
            total_std = jnp.sqrt(
                jnp.mean(gp_var_samples + noise_var_samples, axis=0)
                + jnp.var(means, axis=0)
            )

            y_obs = (data_exp_h[i] if direction == "h" else data_exp_v[i]).flatten()
            resid = y_obs - mu_post
            std_resid = resid / total_std

            for k in range(len(loads)):
                rows.append(
                    {
                        "Angle": angle_deg,
                        "Direction": label,
                        "Load": float(loads[k]),
                        "Residual": float(resid[k]),
                        "StdResidual": float(std_resid[k]),
                    }
                )

            # Dense Grid for Smooth Bands
            dense_loads = jnp.linspace(0, max_load, 100)
            dense_angles = jnp.ones_like(dense_loads) * angle_rad
            dense_xy = jnp.stack([dense_loads, dense_angles], axis=1)

            d_means, d_stds_em = vmap_predict(
                mu_emulator,
                sigma_emulator,
                length_xy,
                length_theta,
                sigma_measure,
                sigma_measure_base,
                test_theta,
                dense_xy,
                direction,
            )

            if noise_model == "additive":
                d_noise_var = (
                    sigma_measure[:, None] ** 2 * dense_loads[None, :]
                    + sigma_measure_base[:, None] ** 2
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


if __name__ == "__main__":
    main()

import numpyro.distributions as dist
import numpy as np

# Default Configuration
config = {
    # Model selection: 'model', 'model_n', 'model_n_hv'
    "model_type": "model_n_hv",
    "seed": 0,  # Random seed for reproducibility
    # Data settings
    "data": {
        # 'noise_model' options: 'proportional' (default), 'additive', or 'constant'
        # 'proportional': sigma^2 = sigma_measure^2 * Load
        # 'additive': sigma^2 = sigma_measure^2 * Load + sigma_base^2
        # 'constant': sigma^2 = sigma_constant^2
        "noise_model": "proportional",
        "base_path": "./data",  # Root data directory
        "max_load": 10.0,  # Maximum load [kN] for data truncation and prediction
        "angles": [45, 90, 135],  # Angles to load
        "prediction_angle": [45, 90, 135],  # Angle for prediction/plotting
        "direction": "v",  # 'h' or 'v' for single direction models/plots
        "prediction_interval": 0.95,  # Prediction interval coverage (e.g., 0.95 for 95%)
        "prediction_samples": 1000,  # Number of samples for prediction
        # Uncertainty bands: "function" (epistemic only), "observation" (includes noise), or "both"
        "uncertainty_bands": "both",
        "run_residual_analysis": True,  # Validation: Run residual analysis
        "plot_trace": True,  # Validation: Plot MCMC trace for diagnostics
    },
    # MCMC settings
    "mcmc": {
        "num_warmup": 3000,
        "num_samples": 3000,
        "num_chains": 2,
        "thinning": 3,
    },
    # Bias flags
    "bias": {
        "add_bias_E1": False,
        "add_bias_alpha": False,
    },
    # Priors
    # Note: These are defined as functions that return numpyro distributions or values
    # This allows them to be instantiated when needed.
    "priors": {
        # Physical parameters (Theta)
        "theta": {
            # For model_n / model_n_hv (reparameterized)
            # These define the transformation: val = mean + scale * standard_normal
            "reparam": {
                "E_1": {"mean": 154900.0, "scale": 5050.0},
                "E_2": {"mean": 10285.0, "scale": 650.0},
                "v_12": {"mean": 0.33, "scale": 0.015},
                "v_23": {"mean": 0.435, "scale": 0.0125},
                "G_12": {"mean": 5115.0, "scale": 98.0},
            },

        },
        # Hyperparameters
        "hyper": {
            # Emulator mean - LogNormal reparameterization: val = exp(log_mean + log_scale * N(0,1))
            "mu_emulator": {"log_mean": -4.605170185988091, "log_scale": 0.1},  # log(0.01) ≈ -4.605
            # Emulator standard deviation
            "sigma_emulator": {
                "target_dist": dist.TruncatedNormal(
                    loc=0.02, scale=0.015, low=0.0
                )
            },
            # Length scales (reparameterized logic)
            # val = exp(mean + scale * standard_normal)
            "length_scales": {
                "lambda_P": {"mean": 3.0, "scale": 0.5},
                "lambda_alpha": {"mean": 0.34, "scale": 0.5},
                "lambda_E1": {"mean": 11.0, "scale": 0.5},
                "lambda_E2": {"mean": 8.3, "scale": 0.5},
                "lambda_v12": {"mean": -0.80, "scale": 0.5},
                "lambda_v23": {"mean": -0.80, "scale": 0.5},
                "lambda_G12": {"mean": 7.7, "scale": 0.5},
            },
            # Measurement noise
            "sigma_measure": {
                "target_dist": dist.TruncatedNormal(
                    loc=0.0001, scale=0.0001, low=0.0
                )
            },
            "sigma_measure_base": {"target_dist": dist.Exponential(100.0)},
            "sigma_constant": {
                "target_dist": dist.Exponential(0.1)
            },  # For constant noise model
        },
        # Bias Priors
        "bias_priors": {
            "sigma_b_E1": dist.Exponential(0.001),
            "sigma_b_alpha": dist.Exponential(1 / np.deg2rad(10)),
        },
    },
}

import numpyro.distributions as dist
import numpy as np

# Default Configuration
config = {
    # Model selection: 'model', 'model_n', 'model_n_hv', 'model_empirical'
    "model_type": "model_empirical",
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
    "empirical": {
        "gamma_scale": 0.8, # SD for gamma_v/h ~ N(cos/sin(alpha), scale)
    },
    # MCMC settings
    "mcmc": {
        "num_warmup": 2000,
        "num_samples": 2000,
        "num_chains": 1,
        "thinning": 2,
    },
    # Bias flags
    "bias": {
        "add_bias_E1": False,
        "add_bias_alpha": False,
        "add_bias_slope": True, # For model_empirical
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
            "mu_emulator": {"log_mean": np.log(0.01), "log_scale": 0.2},  # log(0.01) ≈ -4.605
            # Emulator standard deviation - LogNormal reparameterization
            "sigma_emulator": {"log_mean": np.log(0.03), "log_scale": 0.5},  # ln(0.02) ≈ -3.91
            # Length scales - LogNormal reparameterization: val = exp(log_mean + log_scale * N(0,1))
            "length_scales": {
                "lambda_P": {"log_mean": 3.0, "log_scale": 0.5},
                "lambda_alpha": {"log_mean": 0.34, "log_scale": 0.5},
                "lambda_E1": {"log_mean": 11.0, "log_scale": 0.5},
                "lambda_E2": {"log_mean": 8.3, "log_scale": 0.5},
                "lambda_v12": {"log_mean": -0.80, "log_scale": 0.5},
                "lambda_v23": {"log_mean": -0.80, "log_scale": 0.5},
                "lambda_G12": {"log_mean": 7.7, "log_scale": 0.5},
            },
            # Measurement noise - LogNormal reparameterization
            "sigma_measure": {"log_mean": np.log(0.001), "log_scale": 0.5},  # ln(0.0001) ≈ -9.21
            "sigma_measure_base": {"target_dist": dist.Exponential(100.0)},
            "sigma_constant": {
                "target_dist": dist.Exponential(0.1)
            },  # For constant noise model
        },
        # Bias Priors
        "bias_priors": {
            "sigma_b_E1": dist.Exponential(0.001),
            "sigma_b_alpha": dist.Exponential(1 / np.deg2rad(10)),
            "sigma_b_slope": dist.Exponential(0.001), # For model_empirical
        },
    },
}

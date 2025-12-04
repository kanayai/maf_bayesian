import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as np

# Default Configuration
config = {
    # Model selection: 'model', 'model_n', 'model_n_hv'
    "model_type": "model_n_hv", 
    
    # Data settings
    "data": {
        "base_path": "./data", # Root data directory
        "angles": [45, 90, 135], # Angles to load
        "prediction_angle": 45, # Angle for prediction/plotting
        "direction": "v", # 'h' or 'v' for single direction models/plots
    },

    # MCMC settings
    "mcmc": {
        "num_warmup": 2000,
        "num_samples": 5000,
        "num_chains": 2,
        "thinning": 2,
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
                "E_1":  {"mean": 154900., "scale": 5050.},
                "E_2":  {"mean": 10285.,  "scale": 650.},
                "v_12": {"mean": 0.33,    "scale": 0.015},
                "v_23": {"mean": 0.435,   "scale": 0.0125},
                "G_12": {"mean": 5115.,   "scale": 98.},
            },
            # For standard model (not reparameterized)
            "standard": {
                "E_1":  dist.Normal(161000., 2000.),
                "E_2":  dist.Normal(11380., 100.),
                "v_12": dist.Normal(0.32, 0.01),
                "v_23": dist.Normal(0.43, 0.01),
                "G_12": dist.Normal(5170., 70.),
            }
        },

        # Hyperparameters
        "hyper": {
            # Emulator mean
            "mu_emulator": {
                "mean": 0., 
                "scale": 0.01
            },
            # Emulator standard deviation
            "sigma_emulator": {
                "target_dist": dist.Exponential(20.) # Target distribution for reparam
            },
            # Length scales (reparameterized logic)
            # val = exp(mean + scale * standard_normal)
            "length_scales": {
                "lambda_P":     {"mean": 1.5,   "scale": 0.5},
                "lambda_alpha": {"mean": 0.34,  "scale": 0.5},
                "lambda_E1":    {"mean": 11.,   "scale": 0.5},
                "lambda_E2":    {"mean": 8.3,   "scale": 0.5},
                "lambda_v12":   {"mean": -0.80, "scale": 0.5},
                "lambda_v23":   {"mean": -0.80, "scale": 0.5},
                "lambda_G12":   {"mean": 7.7,   "scale": 0.5},
            },
            # Measurement noise
            "sigma_measure": {
                "target_dist": dist.Exponential(100.)
            }
        },

        # Bias Priors
        "bias_priors": {
            "sigma_b_E1": dist.Exponential(0.0001),
            "sigma_b_alpha": dist.Exponential(1/np.deg2rad(10))
        }
    }
}

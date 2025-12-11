# Analysis Configuration Log

## Reproducibility Info
- **Git Commit**: `59b9dbabc09c1ec91dae634fd1273ed55607a2f0`
- **Results File**: `no_bias_hv_2025_12_10_14_05_16_MAF_linear.nc`

## Configuration Settings
```json
{
    "model_type": "model_n_hv",
    "data": {
        "noise_model": "proportional",
        "base_path": "./data",
        "max_load": 11.0,
        "angles": [
            45,
            90,
            135
        ],
        "prediction_angle": [
            45,
            90,
            135
        ],
        "direction": "v",
        "prediction_interval": 0.95,
        "prediction_samples": 2000,
        "run_residual_analysis": true,
        "plot_trace": true
    },
    "mcmc": {
        "num_warmup": 3000,
        "num_samples": 3000,
        "num_chains": 2,
        "thinning": 3
    },
    "bias": {
        "add_bias_E1": false,
        "add_bias_alpha": false
    },
    "priors": {
        "theta": {
            "reparam": {
                "E_1": {
                    "mean": 154900.0,
                    "scale": 5050.0
                },
                "E_2": {
                    "mean": 10285.0,
                    "scale": 650.0
                },
                "v_12": {
                    "mean": 0.33,
                    "scale": 0.015
                },
                "v_23": {
                    "mean": 0.435,
                    "scale": 0.0125
                },
                "G_12": {
                    "mean": 5115.0,
                    "scale": 98.0
                }
            },
            "standard": {
                "E_1": "<numpyro.distributions.continuous.Normal object at 0x13c0d9010 with batch shape () and event shape ()>",
                "E_2": "<numpyro.distributions.continuous.Normal object at 0x13c108690 with batch shape () and event shape ()>",
                "v_12": "<numpyro.distributions.continuous.Normal object at 0x13c108b90 with batch shape () and event shape ()>",
                "v_23": "<numpyro.distributions.continuous.Normal object at 0x13c0cc770 with batch shape () and event shape ()>",
                "G_12": "<numpyro.distributions.continuous.Normal object at 0x13c0cc9d0 with batch shape () and event shape ()>"
            }
        },
        "hyper": {
            "mu_emulator": {
                "mean": 0.0,
                "scale": 0.01
            },
            "sigma_emulator": {
                "target_dist": "<numpyro.distributions.continuous.Exponential object at 0x13c0d9d30 with batch shape () and event shape ()>"
            },
            "length_scales": {
                "lambda_P": {
                    "mean": 1.5,
                    "scale": 0.5
                },
                "lambda_alpha": {
                    "mean": 0.34,
                    "scale": 0.5
                },
                "lambda_E1": {
                    "mean": 11.0,
                    "scale": 0.5
                },
                "lambda_E2": {
                    "mean": 8.3,
                    "scale": 0.5
                },
                "lambda_v12": {
                    "mean": -0.8,
                    "scale": 0.5
                },
                "lambda_v23": {
                    "mean": -0.8,
                    "scale": 0.5
                },
                "lambda_G12": {
                    "mean": 7.7,
                    "scale": 0.5
                }
            },
            "sigma_measure": {
                "target_dist": "<numpyro.distributions.continuous.Exponential object at 0x13c108cd0 with batch shape () and event shape ()>"
            },
            "sigma_measure_base": {
                "target_dist": "<numpyro.distributions.continuous.Exponential object at 0x13c108e10 with batch shape () and event shape ()>"
            },
            "sigma_constant": {
                "target_dist": "<numpyro.distributions.continuous.Exponential object at 0x13c0cd220 with batch shape () and event shape ()>"
            }
        },
        "bias_priors": {
            "sigma_b_E1": "<numpyro.distributions.continuous.Exponential object at 0x13c0cd0f0 with batch shape () and event shape ()>",
            "sigma_b_alpha": "<numpyro.distributions.continuous.Exponential object at 0x13c0d0dd0 with batch shape () and event shape ()>"
        }
    }
}
```


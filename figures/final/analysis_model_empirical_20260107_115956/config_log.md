# Analysis Configuration Log

## Reproducibility Info
- **Git Commit**: `42b93dcbbdad18a8d1a3a56d2a83fc7803e3e51d`
- **Results File**: `no_bias_normal_45_90_135_2026_01_07_11_57_58_MAF_linear.nc`

## Configuration Settings
```json
{
    "model_type": "model_empirical",
    "seed": 0,
    "data": {
        "noise_model": "proportional",
        "base_path": "./data",
        "max_load": 10.0,
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
        "prediction_samples": 200,
        "uncertainty_bands": "both",
        "run_residual_analysis": true,
        "plot_trace": true
    },
    "empirical": {},
    "mcmc": {
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 1,
        "thinning": 1
    },
    "bias": {
        "add_bias_E1": false,
        "add_bias_alpha": false,
        "add_bias_slope": true
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
            }
        },
        "hyper": {
            "mu_emulator_v": {
                "log_mean": -4.605170185988091,
                "log_scale": 0.2
            },
            "mu_emulator_h": {
                "log_mean": -4.605170185988091,
                "log_scale": 0.2
            },
            "sigma_emulator": {
                "log_mean": -3.506557897319982,
                "log_scale": 0.5
            },
            "length_scales": {
                "lambda_P": {
                    "log_mean": 3.0,
                    "log_scale": 0.5
                },
                "lambda_alpha": {
                    "log_mean": 0.34,
                    "log_scale": 0.5
                },
                "lambda_E1": {
                    "log_mean": 11.0,
                    "log_scale": 0.5
                },
                "lambda_E2": {
                    "log_mean": 8.3,
                    "log_scale": 0.5
                },
                "lambda_v12": {
                    "log_mean": -0.8,
                    "log_scale": 0.5
                },
                "lambda_v23": {
                    "log_mean": -0.8,
                    "log_scale": 0.5
                },
                "lambda_G12": {
                    "log_mean": 7.7,
                    "log_scale": 0.5
                }
            },
            "sigma_measure": {
                "log_mean": -5.298317366548036,
                "log_scale": 0.1
            },
            "sigma_measure_base": {
                "target_dist": "dist.Exponential(100.0)"
            },
            "sigma_constant": {
                "target_dist": "dist.Exponential(0.1)"
            },
            "gamma_scale_v": {
                "target_dist": "dist.Exponential(10.0)"
            },
            "gamma_scale_h": {
                "target_dist": "dist.Exponential(10.0)"
            }
        },
        "bias_priors": {
            "sigma_b_E1": "dist.Exponential(0.001)",
            "sigma_b_alpha": "dist.Exponential(5.729577951308232)",
            "sigma_b_slope": "dist.Exponential(10.0)"
        },
        "simple": {
            "beta_v_45": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            },
            "beta_v_90": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            },
            "beta_v_135": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            },
            "beta_h_45": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            },
            "beta_h_90": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            },
            "beta_h_135": {
                "log_mean": -2.3025850929940455,
                "log_scale": 1
            }
        }
    }
}
```


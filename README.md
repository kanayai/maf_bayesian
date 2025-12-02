# MAF_Bayesian

Bayesian parameter estimation and prediction in multidirectional composite laminates subjected to multiaxial loading.

## Overview
This project uses **Bayesian Inference** to calibrate material properties of composite laminates by combining experimental data with Finite Element (FE) simulation results. It employs a **Gaussian Process (GP) emulator** to correct for potential biases between the simulation and reality.

## Key Features
- **Bayesian Inference**: Uses MCMC (Markov Chain Monte Carlo) via `numpyro` and `jax` to estimate posterior distributions of material parameters ($E_1, E_2, \nu_{12}, G_{12}$).
- **Gaussian Process Emulator**: Bridges the gap between FE models and experimental observations.
- **Bias Correction**: Supports different bias correction frameworks (No bias, Bias $E_1$, Bias $\alpha$).

## Workflow
1.  **Main Execution**: Run `MAF_gp_hv.ipynb` to perform the inference.
    - Loads experimental data:
        - `data_extension_h`: Horizontal extensions (displacement differences).
        - `data_extension_v`: Vertical extensions (displacement differences).
    - Loads simulation data.
    - Runs MCMC sampling.
    - Saves posterior samples to `results_mcmc/`.
2.  **Post-Processing**: Analyze results using the specific notebooks:
    - `posterior_no_bias.ipynb`: For the "No Bias" framework.
    - `posterior_bias_E1.ipynb`: For the "Bias $E_1$" framework.
    - `posterior_bias_alpha.ipynb`: For the "Bias $\alpha$" framework.

## Installation
This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the main script (example)
uv run python MAF_gp_hv.py
```

## Dependencies
- `jax`, `numpyro`: For probabilistic programming.
- `h5py`: For saving/loading results.
- `matplotlib`, `seaborn`: For plotting.
- `pyhelpers`: For saving figures (requires `psycopg2-binary`).

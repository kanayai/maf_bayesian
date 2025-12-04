# MAF_Bayesian

Bayesian parameter estimation and prediction in multidirectional composite laminates subjected to multiaxial loading.

## Overview
This project uses **Bayesian Inference** to calibrate material properties of composite laminates by combining experimental data with Finite Element (FE) simulation results. It employs a **Gaussian Process (GP) emulator** to correct for potential biases between the simulation and reality.

## Key Features
- **Bayesian Inference**: Uses MCMC (Markov Chain Monte Carlo) via `numpyro` and `jax` to estimate posterior distributions of material parameters ($E_1, E_2, \nu_{12}, G_{12}$).
- **Gaussian Process Emulator**: Bridges the gap between FE models and experimental observations.
- **Bias Correction**: Supports different bias correction frameworks (No bias, Bias $E_1$, Bias $\alpha$).
- **Modular Architecture**: Configurable models, priors, and data loading.

## Project Structure

```
maf_bayesian/
├── configs/                 # Configuration files
│   └── default_config.py    # Main configuration (priors, models, settings)
├── src/                     # Source code
│   ├── core/
│   │   ├── models.py        # Bayesian models (refactored)
│   │   └── covariance.py    # Covariance kernels
│   ├── io/
│   │   └── data_loader.py   # Data loading logic
│   └── vis/
│       └── plotting.py      # Plotting functions
├── data/                    # Data directory
│   ├── experimental/        # Experimental data (h/v subfolders)
│   └── simulation/          # Simulation data (h/v subfolders)
├── figures/                 # Output figures from analysis
├── results/                 # Output MCMC results (.nc files)
├── main.py                  # Main entry point for inference
└── analyze.py               # Main entry point for analysis
```

## Models and Likelihoods
The project supports three model types, configurable in `configs/default_config.py`:

1.  **`model`**: Standard formulation using a **Gaussian likelihood**.
2.  **`model_n`**: Reparameterized formulation (improves MCMC mixing) using a **Gaussian likelihood**.
3.  **`model_n_hv`**: Reparameterized formulation using a **Joint Gaussian likelihood** that simultaneously accounts for both Horizontal and Vertical extension data.

## Workflow

### 1. Configuration
Modify `configs/default_config.py` to set:
- **Model**: Choose between `model`, `model_n`, or `model_n_hv`.
- **Priors**: Define priors for physical parameters, hyperparameters, and bias terms using `numpyro.distributions`.
- **Data**: Select angles and data paths.

### 2. Execution
Run the inference pipeline:
```bash
python main.py
```
This script loads data, runs the MCMC sampling based on your config, and saves the posterior samples to `results/`.

### 3. Analysis (`analyze.py`)

The analysis script generates comprehensive visualizations and statistics:

*   **Categorized Posterior Plots**:
    *   **Physical Parameters**: `E_1`, `E_2`, `v_12`, `v_23`, `G_12` (ordered).
    *   **Hyperparameters**: Emulator mean/scale, length scales, measurement noise.
    *   **Normalized Parameters**: `_n` suffixed parameters (standard normal scale).
    *   **Bias Parameters**: If bias is enabled.
    *   *Note*: Plots include the **analytical prior density** (green line) and posterior histogram (density scale).
*   **Prediction Plots**:
    *   **Prior Prediction**: Green dashed intervals.
    *   **Posterior Prediction**: Blue solid intervals.
    *   **Combined Plot**: Data + Prior + Posterior predictions.
*   **Statistics**:
    *   CSV files (`inference_*_stats.csv`) containing Mean, Variance, and Std for all parameters.

To run the analysis:
```bash
uv run python analyze.py
```

## Installation
This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the main script
uv run python main.py
```

## Dependencies
- `jax`, `numpyro`: For probabilistic programming.
- `arviz`: For result storage and analysis.
- `matplotlib`, `seaborn`: For plotting.

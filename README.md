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
├── scripts/                 # Utility scripts (data maintenance, verification)
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
- **Analysis Settings**:
    - `prediction_interval`: Confidence level for prediction plots (default 0.95).
    - `prediction_samples`: Number of samples to use for prediction (default 2000).

### 2. Running Inference

#### **With Automatic Organization (Recommended):**

```bash
# Experimental/testing run → saves to results/tmp/
uv run python main.py --experimental

# Final/important run → saves to results/final/
uv run python main.py --final

# Default run → saves to results/
uv run python main.py
```

Results are automatically saved with timestamped filenames. All `.nc` files are ignored by git and remain local-only.

**Flags:**
- `--experimental`: Automatically saves to `results/tmp/` for quick experiments
- `--final`: Automatically saves to `results/final/` for important runs
- No flag: Saves to `results/` (root level)

### 3. Running Analysis

The analyzer automatically loads the **most recent** `.nc` file from `results/`:

```bash
# Experimental analysis → saves to figures/tmp/
uv run python analyze.py --experimental

# Final analysis → saves to figures/final/
uv run python analyze.py --final

# Default analysis → saves to figures/
uv run python analyze.py
```

Analysis outputs (plots, CSVs) are saved to timestamped directories based on the flag used.

**Flags:**
- `--experimental`: Saves to `figures/tmp/analysis_<timestamp>/`
- `--final`: Saves to `figures/final/analysis_<timestamp>/`
- No flag: Saves to `figures/analysis_<timestamp>/`

#### **Analysis Output**

The analysis script generates comprehensive visualizations and statistics, saved in timestamped folders within `figures/` (e.g., `figures/analysis_model_n_hv_20231027_123045/`).

*   **Categorized Posterior Plots**:
    *   **Physical Parameters**: `E_1`, `E_2`, `v_12`, `v_23`, `G_12` (ordered).
    *   **Hyperparameters**: Emulator mean/scale, length scales, measurement noise.
    *   **Normalized Parameters**: `_n` suffixed parameters (standard normal scale).
    *   **Bias Parameters**: If bias is enabled.
    *   *Note*: Plots include the **analytical prior density** (green line) and posterior histogram (density scale).
*   **Prediction Plots**:
    *   **Prior Prediction**: Green dashed intervals (condition on simulation data only).
    *   **Posterior Prediction**: Blue solid intervals (condition on both simulation and experimental data).
    *   **Dual Direction**: Automatically generates plots for both **Normal** (formerly Vertical) and **Shear** (formerly Horizontal) extension.
    *   **Combined Plot**: Data + Prior + Posterior predictions.
*   **Statistics**:
    *   CSV files (`inference_*_stats.csv`) containing Mean, Variance, and Std for all parameters.

To run the analysis:
```bash
uv run python analyze.py
```

### 4. Reproducibility
To ensure the reproducibility of results, the analysis pipeline automatically generates a **Configuration Log** for every run.

*   **File**: `config_log.md` (saved in the analysis output folder).
*   **Contents**:
    1.  **Git Commit Hash**: The exact version of the code used for the analysis.
    2.  **Results File**: The precise `.nc` file loaded.
    3.  **Configuration Dump**: A full JSON dump of the settings used.

 This allows every figure and statistic to be traced back to the exact code version and dataset that produced it.

## Documentation

For detailed technical documentation, see `docs/`:

- **`prediction_methodology.qmd`**: How prior vs posterior predictions work
- **`prior_prediction_issue.qmd`**: GP theory and Cholesky sampling
- **`posterior_method_explanation.qmd`**: Deep dive into `posterior_predict` function
- **`model_analysis.qmd`**: Model structure and analysis details
- **`residual_analysis.qmd`**: Residual diagnostics

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

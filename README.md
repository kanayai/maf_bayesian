# MAF_Bayesian

Bayesian parameter estimation and prediction in multidirectional composite laminates subjected to multiaxial loading.

## Overview
This project uses **Bayesian Inference** to calibrate material properties of composite laminates by combining experimental data with Finite Element (FE) simulation results. It employs a **Gaussian Process (GP) emulator** to correct for potential biases between the simulation and reality.

## Key Features
- **Bayesian Inference**: Uses MCMC (Markov Chain Monte Carlo) via `numpyro` and `jax` to estimate posterior distributions of material parameters ($E_1, E_2, \nu_{12}, G_{12}$).
- **Gaussian Process Emulator**: Bridges the gap between FE models and experimental observations.
- **Dual Uncertainty Bands**: Prediction plots show both **function uncertainty** (where is the true response?) and **observation uncertainty** (what would a new measurement be?).
- **Flexible Noise Modeling**: Supports Proportional, Additive, and Constant noise models to capture realistic measurement uncertainty.
- **ArviZ Integration**: Uses NetCDF for standardized, robust storage and analysis of Bayesian results.
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
The project supports two active model types (Non-Centered), configurable in `configs/default_config.py`:

1.  **`model_n`**: Reparameterized formulation (improves MCMC mixing) using a **Gaussian likelihood**.
2.  **`model_n_hv`**: Reparameterized formulation using a **Joint Gaussian likelihood** that simultaneously accounts for both Horizontal and Vertical extension data.

## Noise Models
The project supports three noise models to capture measurement uncertainty:

1. **Proportional (Default)**: $\sigma^2 \propto P$. Assumes noise scales with load (heteroscedastic).
2. **Additive**: $\sigma^2 \propto P + \sigma_{base}^2$. Captures background noise at zero load.
3. **Constant**: $\sigma^2 = \text{const}$. Standard homoscedastic noise.

## Workflow

### 1. Configuration
Modify `configs/default_config.py` to set:
- **Model**: Choose between `model_n` or `model_n_hv`.
- **Priors**: Define priors for physical parameters, hyperparameters, and bias terms using `numpyro.distributions`.
- **Data**: Select angles and data paths.
- **Analysis Settings**:
    - `prediction_interval`: Confidence level for prediction plots (default 0.95).
    - `prediction_samples`: Number of samples to use for prediction (default 1000).
    - `uncertainty_bands`: Type of bands to display — `"function"`, `"observation"`, or `"both"` (default).

### 2. Running Inference

#### **With Automatic Organization (Recommended):**

```bash
# Experimental/testing run → saves to results/tmp/<run_id>/
uv run python main.py --experimental

# Experimental run followed by analysis of the exact generated bundle
uv run python scripts/run_experimental_and_analyze.py

# Final/important run → saves to results/final/<run_id>/
uv run python main.py --final

# Default run → saves to results/<run_id>/
uv run python main.py
```

Each inference run now creates an immutable timestamped run bundle containing:

- `manifest.json` with frozen config, status, Git commit/dirty flag, command, seed, and result checksum
- the `.nc` posterior result file

Run bundles are written beneath the selected mode directory. All `.nc` files are ignored by git and remain local-only.

**Flags:**
- `--experimental`: Automatically saves to `results/tmp/<run_id>/` for quick experiments
- `--final`: Automatically saves to `results/final/<run_id>/` for important runs
- No flag: Saves to `results/<run_id>/`

Use `scripts/run_experimental_and_analyze.py` when you want the standard
experimental inference plus matching analysis without manually copying the run
bundle path.

### 3. Running Analysis

The analyzer now requires a verified bundled source so it can use the
manifest-frozen configuration and checksum-checked result rather than mutable
live settings:

```bash
# Experimental analysis from a run bundle directory → saves to figures/tmp/
uv run python analyze.py --results results/tmp/<run_id> --experimental

# Final analysis from a bundled .nc file → saves to figures/final/
uv run python analyze.py --results results/final/<run_id>/<result>.nc --final

# Analysis from a unique run ID → saves to figures/
uv run python analyze.py --results <run_id>
```

Analysis outputs (plots, CSVs) are saved to timestamped directories based on the flag used.

**Flags:**
- `--results PATH_OR_RUN_ID`: Required explicit bundled source: run-bundle directory, bundled `.nc`, or unique run ID
- `--experimental`: Saves to `figures/tmp/analysis_<timestamp>/`
- `--final`: Saves to `figures/final/analysis_<timestamp>/`
- No flag: Saves to `figures/analysis_<timestamp>/`

#### **Analysis Output**

The analysis script generates comprehensive visualizations and statistics, saved in timestamped folders within `figures/` (e.g., `figures/analysis_model_n_hv_20231027_123045/`).

Each analysis output now also includes `exports/` with machine-readable
artefacts tied to the verified source run:

- `analysis_manifest.json`: run ID, source manifest/result, output directory, and exported-file list
- `posterior_summary.csv` and `posterior_summary.json`: posterior summary metrics across analysed parameters
- `diagnostics_summary.json`: sampler diagnostics exported from ArviZ and sample statistics
- `acceptance_summary.json`: explicit phase-5 gate report for divergences, ESS, R-hat availability, and posterior observation coverage
- `prediction_plot_data.csv`: grid-ready prior/posterior prediction intervals and means
- `experimental_observations.csv`: flattened observed experimental points used in prediction plots
- `residual_observations.csv` and `residual_bands.csv`: residual plot-data when residual analysis is enabled

The repository now also tracks a minimal paper evidence registry at
`registry/paper_evidence_registry.json`. Each entry is intended to link one
paper-facing figure, table, or claim to:

- a stable `evidence_id`
- a specific manuscript location
- a source run ID
- a source artefact path
- a source artefact checksum
- a validation status of `candidate` or `accepted`

*   **Categorized Posterior Plots**:
    *   **Physical Parameters**: `E_1`, `E_2`, `v_12`, `v_23`, `G_12` (ordered).
    *   **Hyperparameters**: Emulator mean/scale, length scales, measurement noise.
    *   **Normalized Parameters**: `_n` suffixed parameters (standard normal scale).
    *   **Bias Parameters**: If bias is enabled.
    *   **Derived Slope Parameters**: $\text{slope} = \gamma \times \mu_{\text{emulator}}$ (distributions for each angle/direction).
    *   *Note*: Plots include the **analytical prior density** (green line) and posterior histogram (density scale).
*   **Prediction Plots**:
    *   **Prior Prediction**: Green dashed intervals (condition on simulation data only).
    *   **Posterior Prediction**: Blue solid intervals (condition on both simulation and experimental data).
    *   **Dual Direction**: Automatically generates plots for both **Normal** (formerly Vertical) and **Shear** (formerly Horizontal) extension.
    *   **Combined Plot**: Data + Prior + Posterior predictions.
*   **Statistics**:
    *   CSV files (`inference_*_stats.csv`) containing Mean, Variance, and Std for all parameters.
    *   Structured exports in `exports/` for downstream evidence review and registry insertion.

To run the analysis:
```bash
uv run python analyze.py --results <run_id>
```

### 4. Reproducibility
To ensure the reproducibility of results, the analysis pipeline automatically generates a **Configuration Log** for every run.

*   **File**: `config_log.md` (saved in the analysis output folder).
*   **Contents**:
    1.  **Git Commit Hash**: The exact version of the code used for the analysis.
    2.  **Results File**: The precise `.nc` file loaded.
    3.  **Run ID and Manifest Path**: The verified bundle identity used for analysis.
    4.  **Configuration Dump**: A full JSON dump of the settings used.

 This allows every figure and statistic to be traced back to the exact code version and dataset that produced it.

### 5. Paper Evidence Registry

Validate the tracked registry with:

```bash
uv run python src/io/evidence_registry.py --registry registry/paper_evidence_registry.json --repo-root .
```

The validator rejects missing run IDs, missing artefacts, checksum mismatches,
unknown statuses, duplicate evidence IDs, and ambiguous manuscript locations.

Accepted entries must also include an `acceptance_review` that points to a
matching `acceptance_summary.json` with all phase-5 gates passed. Candidate
entries may carry an acceptance summary for review, but they cannot be promoted
to `accepted` unless that gate report validates successfully.

### 6. Phase-5 pilot

Run the cheap end-to-end pilot with:

```bash
uv run python scripts/run_phase5_pilot.py
```

The pilot creates:

- a new experimental run bundle under `results/tmp/`
- a matching analysis bundle under `figures/tmp/`
- a candidate registry entry in `registry/paper_evidence_registry.json`

The pilot is intentionally cheap and is expected to fail the scientific
acceptance gates. Its purpose is to prove that the full safeguarded provenance
path works and that acceptance remains explicit rather than implicit.

## Documentation

For detailed technical documentation, see `docs/`:

- **`prediction_methodology.qmd`**: How prior vs posterior predictions work
- **`prior_prediction_issue.qmd`**: GP theory and Cholesky sampling
- **`posterior_method_explanation.qmd`**: Deep dive into `posterior_predict` function
- **`model_analysis.qmd`**: Model structure and analysis details
- **`residual_analysis.qmd`**: Residual diagnostics
- **`gp_hyperparameter_analysis.qmd`**: Length scales, uncertainty bands, and noise identifiability

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

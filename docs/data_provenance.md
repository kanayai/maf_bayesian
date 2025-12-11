# Experimental Data Processing

This document attempts to detail the relationship between the raw experimental data and the specific dataset used for Bayesian inference.

Unlike earlier iterations of this project which relied on manual file subsetting, the current workflow employs a **code-driven pipeline** to load, clean, and truncate data dynamically.

## 1. Single Source of Truth

The project relies on one immutable "Full Experimental Dataset":

*   **Location**: `data/experimental/full_experimental_data`
*   **Content**: Complete load-extension curves for all tested angles and directions, extending until failure or test termination.

## 2. Dynamic Processing Pipeline

The `src/io/data_loader.py` script (`load_all_data`) performs the following operations to prepare data for analysis:

### A. Selection (by Config)
It filters files to load only the specific **angles** requested in `defaults_config.py` (e.g., `angles: [45, 90, 135]`).

### B. Cleaning
It enforces data quality rules during loading (`load_experiment_data`):

1.  **NaN Removal**: Any rows containing `NaN` in either Load or Extension are dropped.
2.  **Negative Load Filter**: Rows with significant negative load (tensor compression artifacts < -1e-6) are removed to ensure physical compatibility with the tension-only model inputs.

### C. Truncation (by Load)
To focus the analysis on the relevant elastic/plastic region and avoid high-load failure modes that the model may not capture, the data is **truncated**:

*   **Mechanism**: `truncate_data(..., max_load)`
*   **Configuration**: Controlled by `data.max_load` in `default_config.py`.
*   **Effect**: Any data points where `Load > max_load` are excluded from the inference dataset.

### D. Averaging (for Inference)
As noted in `model_analysis.qmd`, the Bayesian inference currently treats the three sensor positions (Left, Center, Right) as independent samples of the mean behavior.

*   **Operation**: The code computes the mean extension across valid sensors for each load step.
*   **Output**: A single "mean extension" vector per experiment is passed to the likelihood function.

## 3. Reproducibility

Because this processing is defined in code and configuration, the exact dataset used for any analysis run is determined entirely by:

1.  The committed state of `src/io/data_loader.py`.
2.  The values in `configs/default_config.py`.

There is no longer a need to maintain separate manual "partial" data files.

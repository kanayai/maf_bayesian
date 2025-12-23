# Debugging Notes - Bias Plotting Issues
Date: 2025-12-23

## Current Status
The analysis script (`analyze.py`) is running but failing to generate the expected bias plots:
- `posterior_bias_columns_*.png`
- `posterior_bias_n_columns_*.png`
- `inference_bias_stats_*.csv` (partially missing or not updated)

## observations
- The script exits with code 0 (success) in some prior runs, but files are missing.
- In the most recent runs, execution seems to stop or crash silently after generating `posterior_hyper` plots.
- Debug prints have been added to `analyze.py` (lines ~693 and ~767) to trace:
    - `samples_bias` length and keys.
    - `bias_data_by_angle` counts.

## Recent Changes
1.  **`src/vis/plotting.py`**:
    - Implemented `plot_bias_column_layout` for 3-column (45, 90, 135) layout.
    - Fixed `SyntaxWarning` (invalid escape sequences) using raw strings.
    - Added safe casting for `np.percentile` to handle potential JAX types or NaNs.
    - Enforced scientific notation on Y-axis for bias plots.
    - Fixed X-axis range to `(-5, 5)` for bias plots.

2.  **`analyze.py`**:
    - Updated logic to extract `b_{i}_slope` and `b_{i}_slope_n`.
    - Renamed keys to `b_{angle}_{count}` notation.
    - Moved `sigma_measure_n` to hyperparameter plots.
    - Removed duplicate `save_stats_csv` call.

## Next Steps
1.  Run `python3 analyze.py --experimental` and capture **stdout/stderr** to see the debug prints.
2.  If the script hangs, check for infinite loops in the `bias_n_items` sorting or data grouping logic.
3.  Verify if `plot_bias_column_layout` fails silently (e.g. `try/except` masking errors? - checked, no broad except block in that function, but `multi_replace` might have caused indentation issues).
4.  Check `samples_bias_n` extraction: ensure keys like `b_1_slope_n` actually exist in `samples_n`.

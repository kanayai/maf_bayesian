# Handoff — MAF Bayesian plotting variants
_Checkpoint 2026-07-16_

## Objective
Compare prediction-grid presentation variants before deciding whether to keep using the empirical model outputs in the current workflow.

## Done so far
- `plot_grid_spaghetti()` now uses a fixed extension range of `-0.15` to `0.2` mm and a fixed load range of `0` to `15` kN for all prior/posterior grid panels.
- Axis-range-only changes were committed and pushed as `bd8cf3d` (`Adjust spaghetti grid axis ranges`).
- The grid plotting code now supports:
  - `observed_data_mode="average"` or `"raw"`
  - `prediction_mode="function"` or `"observation"`
  - optional function-sample spaghetti via `show_spaghetti`
- `analyze.py` now writes four grid variants per analysis run:
  - `prediction_prior_grid_avg_function_<suffix>.png`
  - `prediction_prior_grid_raw_observation_<suffix>.png`
  - `prediction_posterior_grid_avg_function_<suffix>.png`
  - `prediction_posterior_grid_raw_observation_<suffix>.png`
- Variant behavior currently matches the July 16 request:
  - averaged-data variants plot averaged experimental data plus function-only prediction and keep the faint spaghetti lines
  - raw-data variants plot raw experimental data plus observation-only prediction and suppress spaghetti lines

## Resume point
The code changes are complete but had not yet been exercised through a fresh `analyze.py --results ...` run at handoff time.

## Next action
Run analysis for the intended explicit results bundle, inspect the four new grid outputs, and decide whether the raw-data/observation-only variant needs its own spaghetti or any further legend/text adjustments.

## Last safe commit
`bd8cf3d`

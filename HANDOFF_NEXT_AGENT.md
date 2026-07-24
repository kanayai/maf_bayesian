# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-24 18:00 BST_

## Objective
Test a mechanics-informed empirical slope parameterisation for MAF normal/shear extensions, with shared `mu_emulator` and an explicit shear factor of two.

## Done so far
- Replaced split empirical `mu_emulator_v`/`mu_emulator_h` sampling with one shared `mu_emulator` in `configs/default_config.py` and `src/core/models.py`.
- Added deterministic MCMC outputs `generated_beta_v_i` and `generated_beta_h_i`; analysis now exports/plots generated beta posteriors.
- Added `scripts/plot_empirical_origin_slopes.py`; it fits origin-constrained slopes on averaged data and plots `h/v` extension ratios. Current diagnostics: 45 deg `h/v` median about 1.95, 135 deg about -2.11.
- Updated `docs/empirical_model.qmd` with the scalar `R(alpha)` caveat, DIC normal/shear channel discussion, engineering-shear factor-of-two rationale, and current candidate formulation.
- Baked the shear factor into code: `beta_h = 2.0 * mu_emulator * gamma_h + b_i`; posterior prediction and derived slope analysis now use the same factor.
- Added `scripts/plot_normal_fit_shear_factor_check.py`; it fits only the averaged normal component through the origin for 45 and 135 deg, then predicts shear as `2 * beta_v * tan(alpha)`. Results: 45 deg observed/predicted shear ratio 0.982; 135 deg ratio 1.041.
- Reviewed historical manuscript backups and the `MAF_lhs` FE extraction scripts. No old manuscript/PDF explicitly explains the factor of two. The FE training outputs themselves show shear-vs-`P sin(alpha)` slope about 0.01055 and normal-vs-`P cos(alpha)` slope about 0.00520, ratio 2.03.
- Found a provenance detail: FE `simulation_data_extract.m` applies `disps_corr = (coords_1_corr - coords_0)/2`, while experimental extraction uses `disp_corr = coords_1_corr - coords_0`. This halves both FE directions and does not explain the shear/normal ratio by itself.
- Corrected `docs/direction_specific_emulator.qmd` and `docs/data_provenance.md` to distinguish current code behaviour from diagnostics and interpretation. Current `model_empirical` uses raw three-column DIC arrays as batched replicate curves, while diagnostic scripts may use averaged data.
- Checked `R(alpha)` sensitivity to `E1`: changing `E1` from 154900 MPa to 161000 or 145000 has negligible effect at 45/135 deg; the noticeable effect is near 2 deg only.

## Resume point
Start from branch `feature/empirical-model`. The implementation matches the documented candidate model, but no full MCMC has been run after adding the shear factor and documentation cleanup.

## Next action
Run a small experimental `model_empirical` inference, analyse the explicit run bundle, and inspect `posterior_generated_beta_grid_*`, `posterior_derived_slope_grid_*`, spaghetti grids, and diagnostics before deciding whether this parameterisation is viable.

## Last safe commit
Pending commit from this checkpoint should include only documentation/handoff updates. Leave unrelated local changes in `src/vis/plotting.py` alone unless Karim explicitly asks to commit them.

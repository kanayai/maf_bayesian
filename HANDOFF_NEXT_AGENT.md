# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-23 18:21 BST_

## Objective
Test a mechanics-informed empirical slope parameterisation for MAF normal/shear extensions, with shared `mu_emulator` and an explicit shear factor of two.

## Done so far
- Replaced split empirical `mu_emulator_v`/`mu_emulator_h` sampling with one shared `mu_emulator` in `configs/default_config.py` and `src/core/models.py`.
- Added deterministic MCMC outputs `generated_beta_v_i` and `generated_beta_h_i`; analysis now exports/plots generated beta posteriors.
- Added `scripts/plot_empirical_origin_slopes.py`; it fits origin-constrained slopes on averaged data and plots `h/v` extension ratios. Current diagnostics: 45 deg `h/v` median about 1.95, 135 deg about -2.11.
- Updated `docs/empirical_model.qmd` with the scalar `R(alpha)` caveat, DIC normal/shear channel discussion, engineering-shear factor-of-two rationale, and current candidate formulation.
- Baked the shear factor into code: `beta_h = 2.0 * mu_emulator * gamma_h + b_i`; posterior prediction and derived slope analysis now use the same factor.

## Resume point
Start from commit `bbf002e` on branch `feature/empirical-model`. The implementation matches the documented candidate model, but no full MCMC has been run after adding the shear factor.

## Next action
Run a small experimental `model_empirical` inference, analyse the explicit run bundle, and inspect `posterior_generated_beta_grid_*`, `posterior_derived_slope_grid_*`, spaghetti grids, and diagnostics before deciding whether this parameterisation is viable.

## Last safe commit
`bbf002e` — tree clean before this handoff update.

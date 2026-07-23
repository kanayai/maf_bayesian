# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-23 15:27 BST; updated 2026-07-23 after shared `mu_emulator` edit_

## Objective
Preserve the current empirical-model state before changing the slope parameterisation toward a mechanics-informed shared scale.

## Done so far
- Confirmed `mu_emulator_v` and `mu_emulator_h` use Normal reparameterisations in the config/model path, not LogNormal.
- Updated analysis plots so direction labels read `Normal (v)` and `Shear (h)`, reordered `posterior_hyper`, and included `sigma_b_slope` when slope bias is sampled.
- Added solid green reference lines to the four spaghetti-grid plots and corrected them to plot `Load = Extension / cos(alpha)` for normal and `Load = Extension / sin(alpha)` for shear.
- Documented the mechanics-based empirical slope scale in `docs/empirical_model.qmd`, with sources: `paper/paper.html` Section 2 line 234 and Laux et al. (2020) Figure 4(a)/Table 2 page 5.
- Current config state enables slope bias and uses a shared `mu_emulator` with mean `0.01` and scale `0.1`, `sigma_measure` median `0.0005`, and `sigma_b_slope ~ Exponential(1000)`.
- Replaced the empirical model's split `mu_emulator_v`/`mu_emulator_h` sample sites with one shared `mu_emulator` sample site using the same Normal reparameterisation form.
- Updated `model_empirical` slopes to `beta_v = mu_emulator * gamma_v + b_i` and `beta_h = mu_emulator * gamma_h + b_i`; analysis code now prefers `mu_emulator` while retaining fallback support for historical split-key results.
- Verified with `uv run python -m py_compile configs/default_config.py src/core/models.py analyze.py` and a tiny `Predictive(model_empirical)` prior draw; emitted mu sites were only `mu_emulator` and `mu_emulator_n`.
- Added generated empirical beta outputs: `generated_beta_v_i` and `generated_beta_h_i` are deterministic MCMC sites equal to `mu_emulator * gamma_{v/h,alpha_i} + b_i`. Analysis now plots `posterior_generated_beta_grid_<suffix>.png`, writes `inference_generated_beta_stats_<suffix>.csv`, and includes these sites in posterior summary exports.

## Resume point
Start from a short pilot run of the updated `model_empirical` and inspect whether the shared `mu_emulator`, direction-specific gammas, and generated beta outputs have sensible posterior behaviour.

## Next action
Run a small experimental inference with the updated shared-`mu_emulator` empirical model, then analyse the explicit run bundle if diagnostics are usable.

## Last safe commit
`87fe5e1` before this checkpoint; tree dirty with `configs/default_config.py` and this handoff update pending commit.

# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-23 15:27 BST_

## Objective
Preserve the current empirical-model state before changing the slope parameterisation toward a mechanics-informed shared scale.

## Done so far
- Confirmed `mu_emulator_v` and `mu_emulator_h` use Normal reparameterisations in the config/model path, not LogNormal.
- Updated analysis plots so direction labels read `Normal (v)` and `Shear (h)`, reordered `posterior_hyper`, and included `sigma_b_slope` when slope bias is sampled.
- Added solid green reference lines to the four spaghetti-grid plots and corrected them to plot `Load = Extension / cos(alpha)` for normal and `Load = Extension / sin(alpha)` for shear.
- Documented the mechanics-based empirical slope scale in `docs/empirical_model.qmd`, with sources: `paper/paper.html` Section 2 line 234 and Laux et al. (2020) Figure 4(a)/Table 2 page 5.
- Current uncommitted config state enables slope bias and uses: `mu_emulator_v` scale `0.1`, `mu_emulator_h` scale `0.001`, `sigma_measure` median `0.0005`, and `sigma_b_slope ~ Exponential(1000)`.

## Resume point
Start from `configs/default_config.py` and `src/core/models.py::model_empirical`. Karim wants to try a new empirical model motivated by the finding that the mechanics scale is direction-specific through `R(alpha) cos(alpha)` and `R(alpha) sin(alpha)`, even if the underlying `R(alpha)` comes from a shared compliance idea.

## Next action
Before editing the model, decide the exact new parameterisation: likely replace separate `mu_emulator_v`/`mu_emulator_h` with a shared mechanics-scale parameter (or shared prior mean) and direction factors based on the computed `R(alpha)` table.

## Last safe commit
`87fe5e1` before this checkpoint; tree dirty with `configs/default_config.py` and this handoff update pending commit.

# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-29_

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

## 2026-07-29 session notes
- Clarified units and interpretation: `mu_emulator` has units mm/kN because it maps load directly to extension. It is best interpreted as a global empirical **structural compliance scale**, not a material compliance like `1/E` or `1/G` and not stiffness. Its reciprocal, after choosing the relevant angle/channel transformation, is a rough structural stiffness scale in kN/mm.
- `gamma_v` and `gamma_h` are dimensionless projection factors because they are centred on `cos(alpha)` and `sin(alpha)`.
- The implemented empirical model uses one shared `mu_emulator` that does not depend on loading angle or DIC channel. Angle/channel dependence is carried by `gamma_v(alpha)`, `gamma_h(alpha)`, the shear multiplier, and specimen bias. This is a modelling abstraction, not a material constant.
- Updated `scripts/plot_normal_fit_shear_factor_check.py` to produce six panels: normal/shear for 45, 90, and 135 deg. The 45/135 shear panels still compare observed shear to `2 * beta_v * tan(alpha)` from the normal fit. The 90 deg normal and shear panels are direct origin-constrained least-squares fits, with no factor-of-two imposed.
- Current six-panel diagnostic values: 45 deg `beta_v = 0.003864`, predicted shear `0.007729`, observed shear `0.007590`; 135 deg `beta_v = -0.003786`, predicted shear `0.007573`, observed shear `0.007881`; 90 deg direct fits are `beta_v = -0.000208` and `beta_h = 0.010910` mm/kN.
- Important interpretation update: the experimental diagnostics support a shear factor near two for 45/135 deg, but not directly for 90 deg. Current code still applies `beta_h = 2.0 * mu_emulator * gamma_h + b_i` globally to all angles, including 90 deg. Revisit this before treating the empirical model as final.
- Separated two factor-of-two issues. The FE MATLAB extraction `/2` (`disps_corr = (coords_1_corr - coords_0)/2`) scales both FE normal and shear outputs, so it should **not** be used as evidence for the empirical shear multiplier. It is more plausibly an FE geometry/symmetry/extraction convention, e.g. half-model or half-specimen handling, and needs checking against original FE modelling notes/papers.
- FE simulation data are available in `data/simulation/h` and `data/simulation/v`. Each direction has 100 rows of `input_load_angle_sim.txt` (`P`, `alpha`), 100 rows of `input_theta_sim.txt` (`E_1`, `E_2`, `v_12`, `v_23`, `G_12`), and 100 rows with three FE extension outputs in `data_extension_sim.txt` that the loader averages across columns.
- FE simulation parameter ranges checked this session: `E_1` 141206--166932 MPa, `E_2` 8577--11884 MPa, `v_12` 0.290--0.366, `v_23` 0.404--0.467, `G_12` 4852--5442 MPa; load ranges 0.05--9.98 kN and angles about 0.42--179.31 deg.
- Karim has an idea for using the FE simulation data next. Do not pre-emptively change the QMD files; resume by discussing that idea and deciding whether the empirical `mu_emulator` should be informed by FE-derived structural compliance/stiffness summaries.
- Worktree note: `configs/default_config.py` currently has an unstaged change to the `mu_emulator` prior scale (`0.001` to `0.005`) that was not made by the agent in this handoff step. Confirm with Karim before committing or reverting it.

## Resume point
Start from branch `feature/empirical-model`. The implementation is documented as the current candidate model, but the global shear factor at 90 deg is now flagged as a modelling issue. No full MCMC has been run after adding the shear factor and documentation cleanup.

## Next action
Before running MCMC, discuss Karim's proposed use of the FE simulation data and decide whether the empirical shear multiplier should remain global, be angle-specific, or be replaced by a different FE-informed structural compliance/stiffness parameterisation. Then run a small experimental `model_empirical` inference, analyse the explicit run bundle, and inspect `posterior_generated_beta_grid_*`, `posterior_derived_slope_grid_*`, spaghetti grids, and diagnostics before deciding whether this parameterisation is viable.

## Last safe commit
Pending commit from this checkpoint should include only documentation/handoff updates. Leave unrelated local changes in `src/vis/plotting.py` alone unless Karim explicitly asks to commit them.

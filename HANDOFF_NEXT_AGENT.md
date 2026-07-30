# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-30_

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

## 2026-07-30 session notes
Two threads this session: (A) an FE functional-form exercise, and (B) a brand-new "simplest empirical model". Both are STANDALONE diagnostic scripts — they do not touch the config-driven pipeline or `model_empirical`, and create no paper-evidence run bundles. Both are documented in `docs/empirical_model.qmd`, which is now a properly reproducible Quarto document (executable python chunks regenerate figures/tables on render).

(A) FE functional-form exercise — `scripts/fe_functional_form.py`:
- Purpose: learn a good functional form for the empirical model from the 100-point FE LHS design (controllable P, alpha vs uncontrollable E1,E2,v12,v23,G12). Fits an anisotropic GP surrogate per direction (v=normal, h=shear), standardised inputs, LOO-CV R^2 = 1.000 (FE deterministic → interpolation).
- Findings: extension is linear in load through the origin; normal slope ∝ cos(alpha), shear slope ∝ sin(alpha) (slope/cos and slope/sin constant across angle to ~4% and ~0.1%); only E1 is influential among uncontrollables (E1 acts as a near-uniform ±8%/−6% multiplicative scale; other four inert).
- Shear factor of two: FE supports ~2 at 45/135 (shear/normal = 1.99, 2.09) but it is undefined at 90 (normal ≈ 0). Same caveat as the implemented global factor.
- Cleaner mu_emulator provenance: by construction beta_h(90) = 2*mu, so mu = shear(90)/2 = 0.010612/2 ≈ 0.00531 ≈ config 0.0054. At E1 low/nominal/high, mu = 0.005743 / 0.005306 / 0.004990 (spread well inside the prior scale).
- Figures/CSV (git-ignored) in `figures/fe_functional_form/`: v_vs_angle.png, h_vs_angle.png (two loads P=5,10; axes swapped so extension is horizontal; alpha=45/90/135 drawn as horizontal reference lines), vs_load.png (combined normal-top/shear-bottom), vs_load_with_experiment.png (experiment overlaid; FE lines stop at 10 kN, no extrapolation; experiment softens beyond ~10 kN), vs_angle_nominal_extension.csv (4-row table, ordered by load then direction).
- The 5–95% band = independent UNIFORM draws of each uncontrollable over its FE design range (NOT the priors, NOT the posterior). Documented as such in the QMD.

(B) Simplest empirical model — `scripts/empirical_simple_model.py` (NEW). This is Karim's new baseline model to build on. Decisions (all confirmed by Karim):
- Model, through the origin (no intercept): y_v = P·cos(alpha)·mu + eps ; y_h = 2·P·sin(alpha)·mu + eps. Fixed cos/sin (not random gamma), factor of two imposed exactly, no bias. Single scalar mu, single shared error. v ⟂ h, obs iid.
- Error: PROPORTIONAL model eps ~ Normal(0, sigma^2 · P) (sd = sigma·sqrt(P); band fans with load, zero at origin). Zero-load rows excluded (zero variance).
- Priors: mu ~ Normal(0.0054, 0.001); sigma ~ HalfNormal(0.02) (units mm·kN^−1/2).
- Data: averaged (sensor-mean) experimental, angles 45/90/135, load ≤ 10 kN. 388 obs (normal {45:75,90:50,135:69}, shear same).
- Inference: NUTS, 4 chains × 2000 (1000 warmup), seed 0. Posterior: mu = 0.005429 (sd 1e-5), sigma = 0.000540. Fits shear + normal-45/135 well; normal-90 predicted ≈0 (cos90=0) with proportional band absorbing the small non-zero spread.
- Figure (git-ignored): `figures/empirical_simple_model/predictions_vs_data.png`.

Workflow decision: both empirical models coexist WITHOUT a new branch — the simple model was kept as a standalone script, `model_empirical` untouched. If Karim wants the simple model in the config-driven MCMC pipeline (reproducible run bundles + standard analysis), promote it to a new `model_type = "model_empirical_simple"` in `src/core/models.py` + a config preset. NOT done yet.

Dependency change (flagged): added a `docs` dependency group for Quarto's jupyter engine — `nbformat`, `jupyter-client`, `ipykernel`, `pyyaml`, `nbclient` (in `pyproject.toml`/`uv.lock`). Undo: `uv remove --group docs nbformat jupyter-client ipykernel pyyaml nbclient`. Render command needs the project python: `QUARTO_PYTHON=$(uv run python -c 'import sys;print(sys.executable)') quarto render docs/empirical_model.qmd`.

## Resume point
Start from branch `feature/empirical-model`. The simplest empirical model (proportional error, through-origin, single mu) is fitted and documented as a standalone script; `model_empirical` and `model_n_hv` are untouched. No config-pipeline MCMC / run bundle has been produced for the simple model.

## Next action
Decide with Karim whether to (1) keep iterating on the simplest model and build complexity onto it (his stated plan — "we're gonna start building on over it"), and/or (2) promote it to a config `model_type = "model_empirical_simple"` so it runs through `main.py` with reproducible run bundles and standard analysis/prediction outputs. Then take the next modelling step he specifies.

## Last safe commit
This session's commit includes: `scripts/fe_functional_form.py` (new), `scripts/empirical_simple_model.py` (new), `docs/empirical_model.qmd` (FE + simple-model sections, reproducible chunks), `pyproject.toml`/`uv.lock` (docs jupyter group), and this handoff. Figures and `docs/_site/` are git-ignored and stay local. No changes to `src/` or configs.

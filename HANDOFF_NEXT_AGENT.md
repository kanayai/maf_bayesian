# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-08-04 17:47 BST_

## Objective
Continue the empirical-model/FE-functional-form investigation, especially how
the FE GP surrogate responds to alternative `E1` values from the original
simulation-design spreadsheet.

## Done so far
- Found the FE design workbook: `docs/GP_simulation_design.xlsx`. It reconciles
  `E1` values: Irene `148800`, Hexcel `161000`, combined nominal `154900`, scale
  `5050`; the current paper still states the older Irene prior `148800 (2000)`.
- Added GP 95% posterior predictive intervals to `scripts/fe_functional_form.py`
  load-extension plots; these are very narrow relative to the material-parameter
  design-box envelope.
- Added Irene and Hexcel `E1` mean-GP slices to all FE extension-vs-load plots:
  black = nominal `154900`, purple dashed = Irene `148800`, green dash-dot =
  Hexcel `161000`; `E2`, `v12`, `v23`, and `G12` stay fixed at nominal.
- Re-rendered `docs/empirical_model.qmd`; generated outputs are local/ignored:
  `figures/fe_functional_form/vs_load.png`,
  `figures/fe_functional_form/vs_load_with_experiment.png`, and
  `docs/_site/empirical_model.html`.

## Resume point
Start from `scripts/fe_functional_form.py::plot_vs_load` and
`docs/empirical_model.qmd` around the FE load-extension figure explanation.
Branch is `feature/empirical-model`.

## Next action
Decide whether the empirical model should use the combined FE-design nominal
(`E1 = 154900`) as a sensitivity/reference value only, or whether the manuscript
and/or config should be aligned explicitly with Irene/Hexcel provenance before
the next modelling step.

## Last safe commit
`1930d10` — tree clean before this checkpoint edit.

## Blockers
- Modelling direction remains open: next complexity for the simple empirical
  model has not yet been chosen.

# Handoff — MAF Bayesian paper evidence workflow
_Checkpoint 2026-06-15 17:12_

## Objective
Build a newly generated, explicitly traceable results-to-paper evidence baseline.

## Done so far
- Phases 1-5 of the safeguarded evidence workflow are implemented.
- Analysis now writes `exports/acceptance_summary.json`, and `accepted` registry entries must point to a passing acceptance review.
- A cheap end-to-end pilot was run successfully through inference, analysis, export, and registry validation.
- The pilot artefacts are `results/tmp/model_empirical_20260615T160055698513Z`, `figures/tmp/analysis_model_empirical_20260615_170104`, and registry entry `PILOT-model_empirical_20260615T160055698513Z`.

## Resume point
The workflow is complete; the open decision is whether the current machine acceptance gates are the right scientific baseline after one realistic non-pilot run.

## Next action
Run one non-pilot evidence candidate with realistic MCMC settings, inspect `exports/acceptance_summary.json`, and decide whether to keep or tighten the current thresholds before promoting any entry from `candidate` to `accepted`.

## Last safe commit
`b896c58`

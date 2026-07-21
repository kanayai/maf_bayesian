# Handoff — MAF Bayesian empirical workflow
_Checkpoint 2026-07-21 18:32_

## Objective
Continue empirical-model experimentation with reproducible inference/analysis handoff and inspect the latest empirical analysis outputs.

## Done so far
- Confirmed branch state on `feature/empirical-model`; `paper/paper.qmd` is only on `main`, not the empirical feature branch.
- Committed and pushed Karim's empirical prior change: `7456131` (`mu_emulator_v` and `mu_emulator_h` prior median changed from `0.01` to `0.05`).
- Added, validated, committed, and pushed `scripts/run_experimental_and_analyze.py` as `1e3619a`; command is `uv run python scripts/run_experimental_and_analyze.py`.
- Ran analysis on the latest completed experimental bundle: `results/tmp/model_empirical_20260721T172217019398Z`.
- Latest analysis output: `figures/tmp/analysis_model_empirical_20260721_183052`; worktree was clean after the analysis run.

## Resume point
Start from the latest analysis output in `figures/tmp/analysis_model_empirical_20260721_183052`, especially the posterior/prior grid plots and exported diagnostics. The analysed run bundle is `results/tmp/model_empirical_20260721T172217019398Z`.

## Next action
Open/inspect the latest prediction grid outputs and diagnostics, then decide whether the updated empirical prior scale (`0.05`) improves the Normal/Shear fits enough to keep.

## Last safe commit
`1e3619a` before this checkpoint update; tree clean at handoff start.

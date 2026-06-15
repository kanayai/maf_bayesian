# Handoff — MAF Bayesian paper evidence workflow
_Checkpoint 2026-06-15 14:00_

## Objective
Build a newly generated, explicitly traceable results-to-paper evidence baseline.

## Done so far
- Archived and SHA-256 verified the historical paper and legacy evidence outside
  Git; historical outputs are reference-only.
- Added the concise operational guide in `docs/paper_evidence_workflow.qmd` and
  automatic safeguards in `AGENTS.md`.
- Removed implicit newest-result selection from `analyze.py`.
- Analysis now requires `--results PATH`, validates an existing `.nc` file
  before loading data or creating outputs, and records its resolved path.
- Added `src/io/result_selection.py` and four focused unit tests.
- Implemented immutable inference run bundles in `main.py` via
  `src/io/run_bundle.py`.
- Each run now starts by creating a timestamped bundle directory and
  `manifest.json` with frozen config, status, Git commit/dirty flag, command,
  seed, and placeholder result metadata.
- Successful runs now finish the manifest with the `.nc` SHA-256 checksum and
  file size; failed runs are marked `failed` with the error message.
- Added four focused run-bundle tests and updated the README and paper workflow
  runbook.
- Analysis now resolves an explicit run-bundle directory, bundled `.nc`, or
  unique run ID via `src/io/result_selection.py`.
- Analysis rejects unbundled, incomplete, ambiguous, or checksum-mismatched
  sources before loading data.
- `analyze.py` now uses the manifest-frozen configuration rather than
  `configs/default_config.py`, including reconstructed prior distributions for
  downstream plotting logic.
- `config_log.md` now records the run ID and manifest path for analysed runs.
- Added phase-2 tests for run-ID resolution, checksum verification, bundled
  source enforcement, and frozen-config deserialization.
- Added `src/io/analysis_exports.py` for machine-readable analysis artefacts.
- Each analysis output now writes `exports/analysis_manifest.json`,
  `posterior_summary.csv/json`, `diagnostics_summary.json`,
  `prediction_plot_data.csv`, and `experimental_observations.csv`.
- Residual analysis now also exports `residual_observations.csv` and
  `residual_bands.csv` when enabled.
- Added focused tests for posterior summaries, diagnostics export, prediction
  plot-data flattening, residual export tables, and analysis manifest writing.
- Added the tracked registry file `registry/paper_evidence_registry.json`.
- Added `src/io/evidence_registry.py` with a minimal validator and CLI for
  evidence IDs, manuscript targets, source run IDs, artefact paths, checksums,
  and `candidate`/`accepted` status.
- Added focused tests for registry load/save, checksum validation, duplicate
  evidence IDs, missing artefacts, missing run IDs, bad status values, and
  ambiguous manuscript locations.

## Resume point
Phase 4 is implemented and verified. The next step is to run one cheap
end-to-end pilot that produces a new verified run, analysis exports, at least
one registry entry, and then define practical scientific acceptance gates on
top of that path.

## Next action
Implement phase 5: execute a cheap end-to-end pilot through inference,
analysis, export, and registry validation; then define the minimum scientific
acceptance rules that decide when an entry can move from `candidate` to
`accepted`.

## Karim OS Constraints
- Enforce explicit provenance: never select or analyse a result implicitly.
- Enforce immutability for completed evidence-bearing outputs.
- Keep restartability cheap: each phase must leave an explicit resume point and
  source artefact.
- Keep boundaries clean: code and schemas live in this repo; operational state
  stays in `HANDOFF_NEXT_AGENT.md` and `Karim_AI/karim-ai-os/research/`.

## Remaining phases
1. Minimal immutable run bundle and manifest.
   Karim-rule focus: explicit provenance and no silent overwrite of completed outputs.
2. Analyse by run ID and verify result checksum/frozen config.
   Karim-rule focus: no implicit source selection and no use of mutable live config for evidence-facing analysis.
3. Structured metrics, diagnostics, and plot-data exports.
   Karim-rule focus: evidence artefacts must be explicit, inspectable, and reproducible from a named source run.
4. Minimal paper evidence registry and validator.
   Karim-rule focus: paper-facing claims must have a stable tracked link to accepted source evidence.
5. Cheap end-to-end pilot, then define scientific acceptance rules.
   Karim-rule focus: cold restart must be possible from durable artefacts and documented acceptance gates.

## Verification
- `python3 -m py_compile src/io/evidence_registry.py tests/test_evidence_registry.py` — passed.
- `uv run python -m unittest tests.test_evidence_registry tests.test_analysis_exports tests.test_result_selection tests.test_run_bundle -v` — 29 tests passed.
- `uv run python src/io/evidence_registry.py --help` — documents the minimal validator CLI.
- `quarto render docs/paper_evidence_workflow.qmd` — passed.

## Last safe commit
`53038a1` — structured analysis artefacts; tree dirty with the completed phase-4 evidence-registry changes ready to commit.

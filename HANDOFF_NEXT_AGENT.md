# Handoff — MAF Bayesian paper evidence workflow
_Checkpoint 2026-06-15 13:01_

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

## Resume point
Phase 2 is implemented and verified. The next gap is that analysis outputs are
still mostly human-facing figures and CSVs rather than structured
machine-checkable evidence exports.

## Next action
Implement phase 3: export structured metrics, diagnostics, and plot-data from
the verified source run so downstream evidence review and registry insertion can
use stable machine-readable artefacts.

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
- `python3 -m py_compile analyze.py src/io/result_selection.py src/io/output_manager.py src/io/run_bundle.py tests/test_result_selection.py tests/test_run_bundle.py` — passed.
- `uv run python -m unittest tests.test_result_selection tests.test_run_bundle -v` — 15 tests passed.
- `uv run python -m unittest discover -s tests -v` — 15 tests passed.
- `uv run python analyze.py --help` — documents bundled source selection.
- `uv run python analyze.py` — correctly refuses missing `--results`.
- `quarto render docs/paper_evidence_workflow.qmd` — passed.

## Last safe commit
`4baa53f` — lightweight Karim OS rule overlay; tree dirty with the completed phase-2 analysis-source changes ready to commit.

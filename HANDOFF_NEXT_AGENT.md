# Handoff — MAF Bayesian paper evidence workflow
_Checkpoint 2026-06-15 12:24_

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

## Resume point
Run bundles now exist, but `analyze.py` still takes an explicit `.nc` path and
uses the live config module instead of the manifest's frozen configuration.

## Next action
Implement phase 2: analyse by run ID or bundle path, verify the stored result
checksum before loading, and switch analysis logging to the manifest-frozen
configuration rather than the mutable live config.

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
- `python3 -m py_compile main.py src/io/run_bundle.py src/io/output_manager.py tests/test_run_bundle.py tests/test_result_selection.py` — passed.
- `uv run python -m unittest tests.test_run_bundle tests.test_result_selection -v` — 8 tests passed.
- `uv run python -m unittest discover -s tests -v` — 8 tests passed.
- `quarto render docs/paper_evidence_workflow.qmd` — passed.

## Last safe commit
`7df63fd` — pre-run-bundle baseline; tree dirty with the completed phase ready to commit.

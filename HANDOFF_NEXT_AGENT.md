# Handoff — MAF Bayesian paper evidence workflow
_Checkpoint 2026-06-15 11:16_

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

## Resume point
Explicit result selection is implemented and verified. Run manifests do not yet
exist, so the analysis still uses the current mutable config rather than a
configuration frozen when inference ran.

## Next action
Implement the minimal immutable run bundle and `manifest.json` written by
`main.py`, starting with frozen config, status, Git commit/dirty flag, command,
seed, result checksum, and focused manifest tests.

## Remaining phases
1. Minimal immutable run bundle and manifest.
2. Analyse by run ID and verify result checksum/frozen config.
3. Structured metrics, diagnostics, and plot-data exports.
4. Minimal paper evidence registry and validator.
5. Cheap end-to-end pilot, then define scientific acceptance rules.

## Verification
- `uv run python -m unittest discover -s tests -v` — 4 tests passed.
- `uv run python analyze.py` — correctly refuses missing `--results`.
- `uv run python analyze.py --help` — documents explicit selection.
- `quarto render docs` — passed.

## Last safe commit
`7bc6791` — explicit-result-selection implementation and tests.

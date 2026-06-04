# MAF Bayesian — Agent Orientation

Bayesian parameter estimation and prediction for multidirectional composite
laminates under multiaxial loading. MCMC (`numpyro`/`jax`) with a GP emulator
for FE-vs-experiment bias correction. Full detail: `README.md`.

## Place in the Karim AI system

- This is a **standalone git repo** (`github.com/kanayai/maf_bayesian`), synced to
  all Macs via `sync-infra/repos.txt`. Code lives **here** — never copy it into
  `Karim_AI`. That tree holds only operational state and durable memory.
- Live operational notes and the research resume point for this work live in
  `Karim_AI/karim-ai-os/research/` (`NEXT_STEP.md`, `recovery.md`).
- This location is mapped in `Karim_AI/karim-memory/inventory/locations/maf-bayesian.md`.

## Read first

1. `README.md` — full workflow, models, noise models, reproducibility.
2. `.agent/rules/` — repo conventions (reproducibility, git, docs, roles, output).

## Entry points

- `main.py` — run inference (`uv run python main.py [--experimental|--final]`).
- `analyze.py` — run analysis on the most recent result (large, ~76 KB single file).
- `configs/default_config.py` — models, priors, data selection, analysis settings.
- `src/` — `core/` (models, covariance), `io/` (data, output), `vis/` (plotting).

## Environment

- `uv` for deps, Python 3.13. `uv sync`, then `uv run python …`.

## Known hygiene debt (cleanup deferred, not yet done)

- `.git` is ~312 MB: historical `.h5`/`.nc`/figure blobs committed before
  `.gitignore` caught them. Ignored going forward; still in history.
- `archive/` (~884 files) is old MCMC outputs, mostly supersedable.
- `.nc`/`.h5` results and most figures are git-ignored and stay local-only.
- Do not start a history rewrite or mass deletion without explicit sign-off.

# Decisions

Why key choices were made in this project — the rationale behind the code, not
just the outcome. Newest first. Capture via the `/decision` skill (auto-dated).

## 2026-07-07 — Keep the physics and semi-empirical model branches separate (do not merge yet)

**Context:** The repo has two branches with *unrelated git roots*:
- `main` (Dec 2025) — the **physics-informed** calibration model (`model_n_hv`).
  The engineers' FE-model assumptions are imposed; real physics, but potentially
  biased wherever those assumptions don't hold against the data. This branch
  generated the manuscript's retained results.
- `feature/empirical-model` (Jun 2026) — a deliberately **weakened-assumption /
  semi-empirical** model (`model_empirical`): not fully empirical, but carrying
  *less* physics structure, built to be robust when the physics model is
  misspecified. Also holds the evidence-provenance/registry/run-bundle workflow
  and a direction-specific-emulator refinement.

**Why:** These are two competing *scientific* models, not two versions of one
codebase. They share core machinery but their hyperparameters/priors genuinely
differ — and those differences are unmade modelling decisions. Merging now would
force resolving prior/hyperparameter conflicts by hand, i.e. it would silently
make the paper's model choice before it's been made. The model-strategy decision
(physics / semi-empirical / both) must come first; the merge is downstream of it.

**Provenance constraint:** `main` uniquely holds the commits `ec8563a` and
`02bdcfb` that the manuscript's retained analyses reference (they are *not* on the
feature branch — the roots are unrelated). `main` must be preserved regardless of
what happens to the branches.

**Alternatives:** Rejected (all premature) — full merge with
`--allow-unrelated-histories`, squash-merge onto `main`, and cherry-picking
feature's work onto `main`. Ruled out on technical grounds — `git replace --graft`
and rebasing onto `main` (grafts don't survive clone/sync across the multi-Mac
setup; rebasing 100+ unrelated-root commits is a conflict nightmare).

**Reverse if:** Karim decides the paper's model direction. The intent was always
to eventually merge/consolidate the two branches — once the model strategy is
settled, revisit the merge (base = `main` to preserve provenance).

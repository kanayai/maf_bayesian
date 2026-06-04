# Cleanup Notes

Current hygiene state as of 2026-06-04:

- `.git` is about 312 MB.
- The working tree `archive/` content has been removed completely.
- The main tracked bulk removed in this pass was the full legacy `archive/` tree, including historical outputs, old scripts, and unused data.

Safe cleanup completed in this pass:

- Added ignore rules for `.ipynb_checkpoints/` and `__pycache__/`.
- Added an ignore rule for `archive/` so legacy clutter is not accidentally reintroduced into version control.
- Deleted the full `archive/` tree from the working tree and git index.

Not done in this pass:

- Rewriting git history to shrink `.git`.

Those actions are destructive or history-altering and need an explicit decision first.

Recommended next cleanup step:

1. Decide whether to commit this cleanup on `feature/empirical-model` or move it to a dedicated cleanup branch.
2. Only consider history rewrite afterwards if repository size still matters enough to justify the disruption.

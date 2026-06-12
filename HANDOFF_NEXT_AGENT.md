# Handoff — MAF Bayesian paper
_Checkpoint 2026-06-12_

## Objective
Organise the project for sustained AI-assisted work that finishes the academic paper.

## Done so far
- Audited repository, paper drafts, documentation, branches, results, and activity dates.
- Confirmed latest substantive paper edit was 2025-12-19; latest substantive code change was 2026-01-09.
- Agreed `docs/` should remain in the repo and support the manuscript through an evidence map.
- Aligned the proposed workflow with Karim AI OS: project work stays here; Karim_AI holds thin pointers and reusable knowledge.
- Identified missing exact `.nc` files for three retained analysis folders and branch divergence requiring later review.

## Resume point
Before reorganising anything, protect and inventory the untracked 247 MB `stiffness_paper/` directory.

## Next action
Design and execute a no-loss preservation step for `stiffness_paper/`, including its OneDrive binary archive location and a tracked manifest/pointer.

## Last safe commit
Research-code baseline `94bebd9`; the later checkpoint commit contains only this handoff. Tree dirty only because `stiffness_paper/` remains untracked.

## Blockers
- Do not run `syncwork`: it uses `git add .` and would commit the entire untracked `stiffness_paper/` directory.
- OneDrive target location must be established before moving binary paper artefacts.

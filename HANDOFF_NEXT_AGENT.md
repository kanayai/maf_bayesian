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
- Preserved the untracked `stiffness_paper/` directory in University of Bath
  OneDrive and verified all 87 files by SHA-256 checksum. See
  `PAPER_ARCHIVE.md` and `paper_archive_manifest.csv`.
- Identified `MAF_manuscript_21_NOV_KAI.docx` as the active draft and promoted
  an unchanged copy to the undated OneDrive working file documented in
  `PAPER_WORKING.md`.
- Built the initial manuscript traceability audit in
  `docs/manuscript_evidence_map.qmd` and `docs/manuscript_gaps.qmd`. Confirmed
  that retained no-bias analyses are numerically different from the manuscript
  results and must not be treated as their source.

## Resume point
The paper directory and working manuscript are established, and the initial
evidence map is complete. Recover historical result provenance before changing
manuscript claims or figures.

## Next action
Search collaborator/OneDrive backups for the three exact `.nc` filenames listed
in `docs/manuscript_gaps.qmd`, then locate bias, leave-one-out, sensitivity, and
RMSE/probability-area generating artefacts.

## Last safe commit
Archive preservation commit `371ccef`; working-manuscript pointer and updated
handoff are ready to commit. The local `stiffness_paper/` directory is ignored.

## Blockers
- Exact `.nc` result files corresponding to the three retained analysis folders
  have not yet been located.
- Exact provenance for manuscript Tables 1--11 and Figures 5--17 is unknown.

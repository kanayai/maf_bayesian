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
- Recovered the historical manuscript evidence in the read-only OneDrive
  `MAF_Bayesian-main-old` tree. Exact table values, principal HDF5 links,
  complete bias and leave-one-out families, RMSE/probability-area notebooks,
  and several byte-identical embedded figures now map to the manuscript. See
  `docs/historical_provenance_inventory.qmd`.

## Resume point
The paper directory and working manuscript are established, and most historical
result provenance is recovered. Preserve and canonicalise that evidence before
changing manuscript claims or figures.

## Next action
Create a controlled checksum inventory of the matching non-suffixed legacy
evidence, then decide whether the paper will retain that historical baseline or
adopt a regenerated current-code baseline. Continue searching for the Figure 5
sensitivity source without modifying the legacy tree.

## Last safe commit
Evidence-audit commit `fc70b22`. The local `stiffness_paper/` directory remains
ignored and the OneDrive legacy tree remains untouched.

## Blockers
- Exact `.nc` result files corresponding to three newer retained analysis
  folders have not been located.
- Figure 5 sensitivity provenance and the exact historical generating code
  commit remain unknown.
- Some final aggregate prediction-figure variants still need exact mapping.

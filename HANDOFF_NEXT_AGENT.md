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
- Archived both recovered legacy project trees as reference-only evidence under
  `maf_bayesian_paper_archive/2026-06-15_legacy_evidence/`. Verified 2,935 files
  and 386,127,926 bytes by SHA-256; Git metadata, environments, and caches were
  excluded. See `PAPER_ARCHIVE.md`.
- Added the minimal operational-memory layer for the new workflow:
  `docs/paper_evidence_workflow.qmd` is the concise human runbook and
  `AGENTS.md` contains the non-negotiable agent safeguards.

## Resume point
Historical evidence is preserved and the operational-memory layer is in place.
The preferred direction is a newly generated, explicitly traceable
results-to-paper evidence baseline.

## Next action
Remove implicit newest-result selection from `analyze.py` as the first pipeline
change. Require an explicit result path and add focused selection tests.

## Last safe commit
Legacy-archive commit `d68a83a`; tree has uncommitted operational-memory files.

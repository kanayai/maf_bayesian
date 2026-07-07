# Handoff — next agent / next session

_Checkpoint 2026-07-07 (branch: `main`)_

## Where we are

Two things happened this session: (1) the two-branch situation was diagnosed and a
decision recorded; (2) the paper-writing workflow was scoped into a plan (not yet
executed). Pick up from the **paper workflow plan** below — that is the concrete
next task. The model-strategy decision is Karim's and is not blocking the paper work.

### 1. Branch situation — RESOLVED (decision recorded)

`maf_bayesian` carries two competing models on **unrelated-root git branches**:

- `main` (Dec 2025) — physics-informed `model_n_hv`. Holds manuscript-provenance
  commits `ec8563a`, `02bdcfb` (NOT on the other branch). This is the canonical /
  paper-narrative branch.
- `feature/empirical-model` (Jun 2026) — weakened-assumption `model_empirical` +
  the evidence/provenance/registry workflow + direction-specific emulator.

**Decision (2026-07-07): keep them separate, do NOT merge** until Karim decides the
paper's model (physics / semi-empirical / both). Full rationale in this repo's
`DECISIONS.md` (commit `62af0b7`). Do not "fix" the split by merging.

## 2. Paper workflow — PLAN, ready to execute next session

Goal: Karim drafts his own clean **Quarto** version of the paper (he works in Quarto,
not Word), and only at the very end exports **one-way to `.docx`** for the two
Word-using collaborators. Pattern borrowed from `iquitos_spillover`
(`manuscript/paper.qmd` = single source of truth → `render-and-stage.sh` generates
the collaborator format, never hand-edit the generated file). There the target was
eLife LaTeX → Overleaf; here the target is **Word via pandoc reference-doc**.

### Latest Word source (authoritative, checksummed)

- **`MAF_manuscript_21_NOV_KAI.docx`** — modified **2025-12-19**, Word revision 162,
  41 embedded media, 30 review comments, SHA-256 `d2c62cf1…b07033`. The "21_NOV"
  name is misleading; it is the newest.
- Already promoted to a stable working copy:
  `OneDrive/Mech Eng/OHT data (Tobi Laux)/maf_bayesian_paper/paper/working/maf_bayesian_manuscript.docx`.
- Full checksummed archive: `…/maf_bayesian_paper_archive/2026-06-13/` (87 files).
- Known state: abstract deleted in tracked changes; extensive tracked changes + 30
  comments; reported numbers do NOT reproduce from current code (legacy provenance).

### Execution steps

- **Phase 0 — prerequisites / BLOCKER.** macOS TCC blocks the terminal from reading
  OneDrive (`ls` on the folder → "Operation not permitted"). Before ingest, EITHER
  grant the terminal Full Disk Access, OR (simplest) Karim copies the target `.docx`
  into the repo, e.g. `paper/source/maf_bayesian_manuscript.docx`. Then confirm it
  still matches SHA-256 `d2c62cf1…b07033`.
- **Phase 1 — pandoc ingest, two passes.**
  - Review copy: `pandoc … --track-changes=all --extract-media=media` → captures the
    30 comments, tracked edits, and the deleted abstract as a review checklist.
  - Clean base: `pandoc … --track-changes=accept -t markdown --wrap=none
    --extract-media=media -o paper.qmd`.
  - QA the OMML→LaTeX math; tables → markdown; references likely arrive as plain text
    → plan a `references.bib` rebuild.
- **Phase 2 — restructure into a Quarto skeleton.** YAML front matter (title, authors,
  affils, abstract restored from the review copy), clean section hierarchy, tables,
  `references.bib`. Treat the 41 extracted figures as reference placeholders only
  (figures to be regenerated from code later); cross-map against
  `manuscript_evidence_map.qmd` (on `feature/empirical-model`).
- **Phase 3 — one-way export out.** Quarto `format: docx` with
  `reference-doc: reference.docx` (Bath/journal styling) = the end-of-line, few-times
  export for collaborators; also `html`/`pdf` for Karim. Mirror iquitos's
  `render-and-stage.sh`. Decide how collaborator edits come back (recommend one-way,
  re-enter comments; alt: ingest returned commented `.docx` via `--track-changes`).
- **Phase 4 (later, separate task).** Fold the `docs/*.qmd` methodology documentation
  into the paper workflow. Karim flagged this as the step AFTER the ingest plan.

### Decisions pending from Karim (get these before executing)

1. OneDrive access: grant Full Disk Access, or he drops the `.docx` into the repo.
2. Which branch `paper/` lives on (recommendation: `main`, kept separate from the
   unresolved model decision).
3. Collaborator round-trip: one-way vs ingest-their-returns.

## Last safe commit

`main` @ `62af0b7` (this handoff adds only docs; no code touched this session).

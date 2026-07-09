# Handoff — next agent / next session

_Checkpoint 2026-07-09 (branch: `main`) — supersedes 2026-07-08 on the workflow question_

## UPDATE 2026-07-09 — workflow direction refined (no code changed this session)

New governing split (see `DECISIONS.md` 2026-07-09 entry): **run MCMC on both models and
track every run NOW, on both branches** (physics `main`, semi-empirical
`feature/empirical-model`), forming a curated pool to keep/drop at merge. **Paper-wiring
stays deferred** to single-branch convergence — the wire-early-against-`main` idea was
rejected (front-loads expensive model-specific work against a model that might be dropped).

**Concrete next action:** inspect the "evidence-provenance/registry/run-bundle workflow" said
to already live on `feature/empirical-model`, and what `main`'s `analyze.py` currently emits.
The likely task is **porting that run-tracking onto `main` / making it cross-branch**, not
building new. Nothing was implemented this session — this is a decision-only checkpoint.

The Phase 1–3 roadmap below still holds; only the run-tracking timing moved earlier.

## TL;DR — where we are

The paper is being authored in **Quarto** (`paper/paper.qmd`) as a **living document**.
Phases 0–2 of the Word→Quarto migration are **done and committed**: source ingested, a
full Quarto skeleton built, renders cleanly to HTML. The project is now **parked at a
deliberate boundary**: Karim goes off to run **lots of MCMC on both models** to decide the
final model. **No paper-reproducibility wiring happens until he converges to one model on
one branch.** Do not wire figures/tables early.

## The governing plan (read `DECISIONS.md` 2026-07-08 entries for full rationale)

The paper co-evolves with the science. Two kinds of change, kept separate:

| Kind | Driver | Mechanism |
|---|---|---|
| **Narrative** (prose, structure, argument, which model) | Karim authoring | hand-edits `paper.qmd`, ongoing |
| **Results** (figures, tables, numbers) | model code | code **generates** artifacts; `.qmd` **links** figures / **reads** CSVs; rerun → re-render → auto-updates. **No manual copy.** |

Rules: heavy MCMC/GP code stays **out** of the `.qmd` (only lightweight read/format glue in
the paper); each run emits a `manifest.json` for provenance; freeze a checksummed bundle at
submission.

### Phased roadmap
- **Phase 1 — Exploration (NOW).** Karim runs MCMC on both models (physics on `main`,
  semi-empirical on `feature/empirical-model`). Paper is a provisional living draft; pasted
  images are temporary placeholders only. **No wiring.**
- **Milestone — Convergence.** Karim settles on the final model (physics / semi-empirical /
  combination) → branches collapse to **one branch, one model**.
- **Phase 2 — Wire (once, at convergence).** Define the artifact contract against the real
  converged model (figure names/formats PDF|SVG, tables CSV|JSON, output dir, `manifest.json`);
  update that model's analysis code to emit them; rewire `paper.qmd` to link/read them and
  delete the placeholder images; add `render-and-stage.sh` (regenerate → `quarto render` →
  HTML/PDF for Karim, one-way `.docx` for collaborators).
- **Phase 3 — Steady state.** Rerun model → figures/tables auto-update on re-render; Karim
  keeps evolving prose; freeze provenance bundle at submission.

**Do NOT** rebuild the earlier cross-branch/model-partitioned-outputs machinery — it was
ruled unnecessary because convergence to one branch precedes wiring.

## What was done this session (all committed except final handoff)

- **Phase 0** — copied the authoritative OneDrive working copy into `paper/source/`
  (`maf_bayesian_manuscript.docx`, git-ignored); **SHA-256 verified** = `d2c62cf1…b07033`
  (matches the promoted `21_NOV_KAI` source). TCC did NOT block the terminal this session.
- **Phase 1** — two-pass pandoc ingest (Quarto's bundled pandoc 3.8.3; no standalone pandoc
  installed): `paper/paper_ingest.md` (clean base) + `paper/review/manuscript_review.md`
  (all tracked changes + 16 comments; abstract recovered from here).
- **Phase 2** — `paper/paper.qmd`: YAML (4 authors, Bath/Bristol affils, restored abstract,
  keywords, `references.bib`, commented `docx` export stub); full §1–5 heading hierarchy;
  14 labelled equations `@eq-1..14`; 11 tables transcribed to Markdown; 17 figure
  placeholders `@fig-*`; all `[N]` → `[@refN]`. **Renders to HTML with zero unresolved
  xrefs/citations.** QA checklist is at the top of `paper.qmd`.
- `paper/references.bib` — all 42 refs, keys `ref1..ref42` = manuscript `[1]..[42]`.
- Decisions recorded in `DECISIONS.md`: (1) reproducibility/wiring plan, (2) model-deferral +
  paper-track kickoff, (3) one-way collaborator round-trip (Option A), (4) keep branches
  separate.
- Housekeeping: un-tracked `paper/paper.html` + `paper/paper_files/` (generated) and added
  them to `.gitignore`.

## Known issues / QA debt (do NOT lose these)

- **Reported numbers do NOT reproduce from current code** (legacy provenance) — every number
  and figure in `paper.qmd` is a **placeholder**, to be regenerated in Phase 2. See the QA
  checklist atop `paper.qmd`.
- `fig-loo-be1` source is a Windows `.emf` — will not render; regenerate from code.
- Transcribed table cells must be double-checked against source before submission.
- Two author queries preserved as a `callout-note` in §5; one in-text TODO ("Add more recent
  references from CMAME journal").

## Decisions pending from Karim (deferred, not blocking Phase 1)

1. **Final model choice** — the gating decision; comes out of the MCMC exploration.
2. Canonical output location for artifacts (recommended: in-repo `results/paper_artifacts/`,
   or a sibling dir outside the repo) — decide at wiring time.

## Last safe commit

`main` — this handoff + `DECISIONS.md` update + gitignore/html-untrack are the only changes
since the Phase 0–2 commit `92f02da`. No model code touched.

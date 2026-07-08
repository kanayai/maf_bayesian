# paper/source — Word ingest source

Drop the **active manuscript working copy** here as `maf_bayesian_manuscript.docx`:

    OneDrive: Mech Eng/OHT data (Tobi Laux)/maf_bayesian_paper/paper/working/maf_bayesian_manuscript.docx

This is the authoritative ingest source for the Word→Quarto migration (Phase 0 of
`HANDOFF_NEXT_AGENT.md`). It was promoted from `MAF_manuscript_21_NOV_KAI.docx`
(2025-12-19). The terminal cannot read OneDrive directly (macOS TCC), so copy it in
manually.

After copying, verify the checksum matches the promoted source:

    shasum -a 256 paper/source/maf_bayesian_manuscript.docx
    # expect: d2c62cf1…b07033  (if it differs, the working copy has diverged)

The `.docx` itself is git-ignored (binary; provenance lives in the OneDrive archive).
Only this README is tracked.

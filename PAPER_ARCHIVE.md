# MAF paper archive

The untracked paper working directory was preserved on 2026-06-13 before
reorganising the repository.

## Archive location

University of Bath OneDrive:

`Mech Eng/OHT data (Tobi Laux)/maf_bayesian_paper_archive/2026-06-13/`

The archive contains:

- `stiffness_paper/`: an unchanged copy of the local working directory
- `paper_archive_manifest.csv`: a snapshot of the tracked checksum manifest

The original local `stiffness_paper/` directory remains in place and is ignored
by Git.

## Verification

- Files: 87 source and 87 archive
- Total bytes: 259,132,375
- Content verification: every relative path, byte size, and SHA-256 checksum
  matches
- Timestamp note: OneDrive removed sub-second precision from the modification
  times of the two `.DS_Store` files; their sizes and checksums match

The tracked [`paper_archive_manifest.csv`](paper_archive_manifest.csv) records
each file's relative path, category, size, modification time, and SHA-256
checksum. Regenerate it with:

```bash
python3 scripts/create_paper_archive_manifest.py \
  stiffness_paper paper_archive_manifest.csv
```

## Inventory summary

| Category | Files | Bytes |
|---|---:|---:|
| Manuscripts or working documents | 27 | 211,442,198 |
| Analysis outputs | 51 | 11,966,709 |
| Reference or manuscript PDFs | 5 | 15,184,779 |
| Presentation | 1 | 20,506,762 |
| Table or data | 1 | 15,535 |
| Other (`.DS_Store`) | 2 | 16,392 |

The latest manuscript by modification time is
`MAF_manuscript_21_NOV_KAI.docx`, modified on 2025-12-19.

## Legacy evidence archive

The two recovered historical project trees were separately preserved on
2026-06-15 before starting a new results-to-paper evidence baseline.

University of Bath OneDrive:

`Mech Eng/OHT data (Tobi Laux)/maf_bayesian_paper_archive/2026-06-15_legacy_evidence/`

Archived source trees:

- `MAF_Bayesian-main-old/`
- `MAF_Bayesian-main-backup-1-dec-2025/`

The archive contains 2,935 research-evidence files totalling 386,127,926 bytes.
Every copied file matched its source byte size and SHA-256 checksum.

The archive deliberately excludes `.git`, `.conda`, `.venv`, `__pycache__`,
`.ipynb_checkpoints`, and `.DS_Store` content. These are repository metadata,
recreatable environments, or caches rather than historical research evidence.

The external archive includes:

- `legacy_archive_manifest.csv`: path, size, modification time, and SHA-256 for
  every archived file;
- `legacy_archive_summary.json`: source trees, exclusions, totals, and
  verification status.

Regenerate a new dated legacy archive with
`scripts/archive_legacy_evidence.py`. The script refuses to overwrite an
existing archive destination and marks incomplete archives on failure.

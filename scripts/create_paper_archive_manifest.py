"""Create a checksum manifest for the external paper archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if any(part.startswith("analysis_model_") for part in path.parts):
        return "analysis_output"
    if suffix == ".docx":
        return "manuscript_or_working_document"
    if suffix == ".pdf":
        return "reference_or_manuscript_pdf"
    if suffix in {".csv", ".xlsx"}:
        return "table_or_data"
    if suffix == ".pptx":
        return "presentation"
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    files = sorted(path for path in source.rglob("*") if path.is_file())

    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            ["relative_path", "category", "bytes", "modified_at", "sha256"]
        )
        for path in files:
            stat = path.stat()
            writer.writerow(
                [
                    path.relative_to(source),
                    classify(path.relative_to(source)),
                    stat.st_size,
                    datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                    sha256(path),
                ]
            )


if __name__ == "__main__":
    main()

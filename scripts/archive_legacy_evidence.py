"""Copy legacy evidence trees to a verified external archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    ".conda",
    ".git",
    ".ipynb_checkpoints",
    ".venv",
    "__pycache__",
}
EXCLUDED_FILES = {".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_files(source: Path) -> Iterator[Path]:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_file() and path.name not in EXCLUDED_FILES:
            yield path


def copy_tree(source: Path, destination: Path, tree_name: str) -> list[dict[str, object]]:
    records = []
    for source_path in evidence_files(source):
        relative = source_path.relative_to(source)
        destination_path = destination / tree_name / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

        source_stat = source_path.stat()
        source_hash = sha256(source_path)
        destination_hash = sha256(destination_path)
        if source_stat.st_size != destination_path.stat().st_size:
            raise RuntimeError(f"Size mismatch after copy: {source_path}")
        if source_hash != destination_hash:
            raise RuntimeError(f"Checksum mismatch after copy: {source_path}")

        records.append(
            {
                "tree": tree_name,
                "relative_path": str(relative),
                "bytes": source_stat.st_size,
                "modified_at": datetime.fromtimestamp(source_stat.st_mtime)
                .astimezone()
                .isoformat(),
                "sha256": source_hash,
            }
        )
    return records


def write_manifest(records: list[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["tree", "relative_path", "bytes", "modified_at", "sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()

    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    destination.mkdir(parents=True)

    records: list[dict[str, object]] = []
    try:
        for source in args.sources:
            source = source.resolve()
            if not source.is_dir():
                raise NotADirectoryError(source)
            records.extend(copy_tree(source, destination, source.name))

        write_manifest(records, destination / "legacy_archive_manifest.csv")
        summary = {
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "verified",
            "file_count": len(records),
            "total_bytes": sum(int(record["bytes"]) for record in records),
            "source_trees": [str(source.resolve()) for source in args.sources],
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
            "excluded_files": sorted(EXCLUDED_FILES),
            "verification": "Every copied file matched its source byte size and SHA-256.",
        }
        with (destination / "legacy_archive_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
            file.write("\n")
    except Exception:
        marker = destination / "ARCHIVE_INCOMPLETE"
        marker.write_text(
            "Archive creation or verification failed. Do not treat as verified.\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.io.run_bundle import sha256_file


DEFAULT_REGISTRY_PATH = Path("registry") / "paper_evidence_registry.json"
ALLOWED_STATUSES = {"candidate", "accepted"}
MANUSCRIPT_LOCATION_PREFIXES = (
    "Figure ",
    "Table ",
    "Section ",
    "Equation ",
    "Claim ",
    "Appendix ",
)


class RegistryValidationError(ValueError):
    """Raised when the paper evidence registry fails validation."""


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    return json.loads(registry_path.read_text())


def save_registry(document: dict[str, Any], path: Path | str = DEFAULT_REGISTRY_PATH) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(document, indent=2) + "\n")
    return registry_path


def validate_registry_document(
    document: dict[str, Any],
    *,
    repo_root: Path | str,
) -> list[dict[str, Any]]:
    if document.get("schema_version") != 1:
        raise RegistryValidationError("Registry schema_version must be 1")

    entries = document.get("entries")
    if not isinstance(entries, list):
        raise RegistryValidationError("Registry entries must be a list")

    validated_entries = []
    seen_ids: set[str] = set()
    root = Path(repo_root).resolve()

    for entry in entries:
        validated_entries.append(validate_registry_entry(entry, seen_ids=seen_ids, repo_root=root))

    return validated_entries


def validate_registry_file(
    path: Path | str = DEFAULT_REGISTRY_PATH,
    *,
    repo_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    registry_path = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else registry_path.parent.parent.resolve()
    return validate_registry_document(load_registry(registry_path), repo_root=root)


def validate_registry_entry(
    entry: dict[str, Any],
    *,
    seen_ids: set[str],
    repo_root: Path,
) -> dict[str, Any]:
    required_fields = [
        "evidence_id",
        "manuscript_location",
        "source_run_id",
        "source_artifact",
        "source_artifact_sha256",
        "status",
    ]

    for field in required_fields:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RegistryValidationError(f"Registry entry is missing required field: {field}")

    evidence_id = entry["evidence_id"].strip()
    if evidence_id in seen_ids:
        raise RegistryValidationError(f"Duplicate evidence_id in registry: {evidence_id}")
    seen_ids.add(evidence_id)

    manuscript_location = entry["manuscript_location"].strip()
    if not manuscript_location.startswith(MANUSCRIPT_LOCATION_PREFIXES):
        allowed = ", ".join(prefix.strip() for prefix in MANUSCRIPT_LOCATION_PREFIXES)
        raise RegistryValidationError(
            "Registry manuscript_location must start with a specific manuscript target "
            f"({allowed}): {manuscript_location}"
        )

    status = entry["status"].strip()
    if status not in ALLOWED_STATUSES:
        raise RegistryValidationError(
            f"Registry status must be one of {sorted(ALLOWED_STATUSES)}: {status}"
        )

    artifact_path = resolve_registry_artifact(entry["source_artifact"], repo_root=repo_root)
    if not artifact_path.exists():
        raise RegistryValidationError(f"Registry source_artifact does not exist: {artifact_path}")
    if not artifact_path.is_file():
        raise RegistryValidationError(f"Registry source_artifact is not a file: {artifact_path}")

    expected_sha = entry["source_artifact_sha256"].strip()
    actual_sha = sha256_file(artifact_path)
    if actual_sha != expected_sha:
        raise RegistryValidationError(
            "Registry source_artifact checksum mismatch for "
            f"{artifact_path}: expected {expected_sha}, got {actual_sha}"
        )

    normalized = dict(entry)
    normalized["evidence_id"] = evidence_id
    normalized["manuscript_location"] = manuscript_location
    normalized["status"] = status
    normalized["resolved_source_artifact"] = str(artifact_path)
    return normalized


def resolve_registry_artifact(source_artifact: str, *, repo_root: Path) -> Path:
    artifact_path = Path(source_artifact).expanduser()
    if artifact_path.is_absolute():
        return artifact_path.resolve()
    return (repo_root / artifact_path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the paper evidence registry against required metadata and source artefacts.",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to the paper evidence registry JSON file.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve relative source artefact paths.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        entries = validate_registry_file(args.registry, repo_root=args.repo_root)
    except RegistryValidationError as exc:
        raise SystemExit(f"Registry validation failed: {exc}") from exc

    print(f"Registry validation passed: {len(entries)} entries checked.")


if __name__ == "__main__":
    main()

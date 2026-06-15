"""Explicit analysis source selection for bundled inference results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.io.output_manager import deserialize_config
from src.io.run_bundle import sha256_file


@dataclass(frozen=True)
class AnalysisSource:
    run_id: str
    bundle_dir: Path
    manifest_path: Path
    result_path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]


def existing_analysis_source(value: str) -> AnalysisSource:
    """
    Resolve and validate an explicitly selected analysis source.

    Accepted inputs:
    - a run-bundle directory containing ``manifest.json``
    - a bundled ``.nc`` result file whose parent contains ``manifest.json``
    - a run ID that resolves uniquely under ``results/``
    """
    candidate = Path(value).expanduser()

    if candidate.exists():
        resolved = candidate.resolve()
        if resolved.is_dir():
            return _analysis_source_from_bundle_dir(resolved)
        if resolved.is_file():
            return _analysis_source_from_result_file(resolved)
        raise argparse.ArgumentTypeError(f"Analysis source is neither file nor directory: {resolved}")

    return _analysis_source_from_run_id(value)


def existing_netcdf_path(value: str) -> Path:
    """
    Legacy explicit-NetCDF validator retained for backwards-compatible tests.

    New analysis code should prefer ``existing_analysis_source``.
    """
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Result file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Result path is not a file: {path}")
    if path.suffix.lower() != ".nc":
        raise argparse.ArgumentTypeError(f"Result file must use the .nc extension: {path}")
    return path


def _analysis_source_from_result_file(path: Path) -> AnalysisSource:
    if path.suffix.lower() != ".nc":
        raise argparse.ArgumentTypeError(f"Result file must use the .nc extension: {path}")
    return _analysis_source_from_bundle_dir(path.parent, explicit_result_path=path)


def _analysis_source_from_bundle_dir(
    bundle_dir: Path,
    explicit_result_path: Path | None = None,
) -> AnalysisSource:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise argparse.ArgumentTypeError(
            f"Run bundle manifest is missing: {manifest_path}. "
            "Analysis now requires a bundled result with manifest.json."
        )
    if not manifest_path.is_file():
        raise argparse.ArgumentTypeError(f"Run bundle manifest is not a file: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Run bundle manifest is not valid JSON: {manifest_path}") from exc

    if manifest.get("status") != "completed":
        raise argparse.ArgumentTypeError(
            f"Run bundle is not completed and cannot be analysed: {bundle_dir}"
        )

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise argparse.ArgumentTypeError(f"Run bundle manifest is missing a valid run_id: {manifest_path}")

    result_info = manifest.get("result")
    if not isinstance(result_info, dict):
        raise argparse.ArgumentTypeError(f"Run bundle manifest is missing result metadata: {manifest_path}")

    result_name = result_info.get("path")
    if not isinstance(result_name, str) or not result_name:
        raise argparse.ArgumentTypeError(f"Run bundle manifest is missing result path: {manifest_path}")

    result_path = (bundle_dir / result_name).resolve()
    if explicit_result_path is not None and result_path != explicit_result_path.resolve():
        raise argparse.ArgumentTypeError(
            "Selected .nc file does not match the manifest-declared bundled result: "
            f"{explicit_result_path}"
        )
    if not result_path.exists():
        raise argparse.ArgumentTypeError(f"Bundled result file does not exist: {result_path}")
    if not result_path.is_file():
        raise argparse.ArgumentTypeError(f"Bundled result path is not a file: {result_path}")
    if result_path.suffix.lower() != ".nc":
        raise argparse.ArgumentTypeError(f"Bundled result must use the .nc extension: {result_path}")

    expected_sha = result_info.get("sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise argparse.ArgumentTypeError(f"Run bundle manifest is missing result SHA-256: {manifest_path}")

    actual_sha = sha256_file(result_path)
    if actual_sha != expected_sha:
        raise argparse.ArgumentTypeError(
            f"Bundled result checksum mismatch for {result_path}: expected {expected_sha}, got {actual_sha}"
        )

    serialized_config = manifest.get("config")
    if not isinstance(serialized_config, dict):
        raise argparse.ArgumentTypeError(f"Run bundle manifest is missing frozen config: {manifest_path}")
    config = deserialize_config(serialized_config)

    return AnalysisSource(
        run_id=run_id,
        bundle_dir=bundle_dir.resolve(),
        manifest_path=manifest_path.resolve(),
        result_path=result_path,
        manifest=manifest,
        config=config,
    )


def _analysis_source_from_run_id(run_id: str) -> AnalysisSource:
    search_roots = [
        Path("results"),
        Path("results") / "tmp",
        Path("results") / "final",
    ]

    matches = []
    for root in search_roots:
        candidate = (root / run_id).resolve()
        if candidate.exists() and candidate.is_dir():
            matches.append(candidate)

    if not matches:
        raise argparse.ArgumentTypeError(
            "Analysis source was not found as a path or run ID under results/: "
            f"{run_id}"
        )
    if len(matches) > 1:
        raise argparse.ArgumentTypeError(
            "Run ID is ambiguous across results roots; pass the bundle path explicitly: "
            f"{run_id}"
        )

    return _analysis_source_from_bundle_dir(matches[0])

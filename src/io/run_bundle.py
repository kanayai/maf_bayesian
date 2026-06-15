from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.io.output_manager import _config_encoder


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    bundle_dir: Path
    result_path: Path
    manifest_path: Path


def result_output_root(output_mode: str) -> Path:
    if output_mode == "experimental":
        return Path("results") / "tmp"
    if output_mode == "final":
        return Path("results") / "final"
    return Path("results")


def build_result_filename(config: dict[str, Any], timestamp: str) -> str:
    angles = config["data"]["angles"]
    model_type = config["model_type"]

    if model_type == "model_n_hv":
        suffix = "hv" + "".join([f"_{i}" for i in angles]) if len(angles) != 3 else "hv"
    else:
        direction = config["data"].get("direction", "h")
        dir_tag = "shear" if direction == "h" else "normal"
        suffix = f"{dir_tag}" + "".join([f"_{i}" for i in angles])

    bias_flags = config["bias"]
    prefix = "bias_" if (bias_flags["add_bias_E1"] or bias_flags["add_bias_alpha"]) else "no_bias_"
    if bias_flags["add_bias_E1"]:
        prefix += "E1_"
    if bias_flags["add_bias_alpha"]:
        prefix += "alpha_"

    return f"{prefix}{suffix}_{timestamp}_MAF_linear.nc"


def prepare_run_bundle(
    config: dict[str, Any],
    output_mode: str,
    command: list[str],
) -> RunBundle:
    output_root = result_output_root(output_mode)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{config['model_type']}_{timestamp}"
    bundle_dir = output_root / run_id
    bundle_dir.mkdir(parents=False, exist_ok=False)

    result_path = bundle_dir / build_result_filename(config, timestamp)
    manifest_path = bundle_dir / "manifest.json"

    bundle = RunBundle(
        run_id=run_id,
        bundle_dir=bundle_dir,
        result_path=result_path,
        manifest_path=manifest_path,
    )

    manifest = _base_manifest(bundle, config, output_mode, command)
    _write_manifest(manifest_path, manifest)
    return bundle


def mark_run_completed(bundle: RunBundle) -> dict[str, Any]:
    manifest = _read_manifest(bundle.manifest_path)
    checksum = sha256_file(bundle.result_path)
    stat = bundle.result_path.stat()

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = utc_now_iso()
    manifest["result"] = {
        "path": bundle.result_path.name,
        "sha256": checksum,
        "size_bytes": stat.st_size,
    }

    _write_manifest(bundle.manifest_path, manifest)
    return manifest


def mark_run_failed(bundle: RunBundle, error_message: str) -> dict[str, Any]:
    manifest = _read_manifest(bundle.manifest_path)
    manifest["status"] = "failed"
    manifest["completed_at_utc"] = utc_now_iso()
    manifest["error"] = error_message
    _write_manifest(bundle.manifest_path, manifest)
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _base_manifest(
    bundle: RunBundle,
    config: dict[str, Any],
    output_mode: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": bundle.run_id,
        "status": "running",
        "created_at_utc": utc_now_iso(),
        "completed_at_utc": None,
        "output_mode": output_mode,
        "model_type": config["model_type"],
        "seed": config.get("seed", 0),
        "command": command,
        "bundle_dir": str(bundle.bundle_dir),
        "git": git_metadata(),
        "result": {
            "path": bundle.result_path.name,
            "sha256": None,
            "size_bytes": None,
        },
        "config": json.loads(json.dumps(config, default=_config_encoder)),
    }


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
        )
        return {"commit": commit, "is_dirty": dirty}
    except Exception:
        return {"commit": None, "is_dirty": None}


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

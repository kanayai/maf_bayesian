from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.default_config import config as default_config
from main import run_inference, save_results
from src.core.models import model_empirical, model_n, model_n_hv, model_simple
from src.io.evidence_registry import load_registry, save_registry, validate_registry_file
from src.io.run_bundle import mark_run_failed, prepare_run_bundle, sha256_file

import jax.random as random
from src.io.data_loader import load_all_data


def pilot_config() -> dict:
    config = copy.deepcopy(default_config)
    config["mcmc"]["num_warmup"] = 25
    config["mcmc"]["num_samples"] = 25
    config["mcmc"]["num_chains"] = 1
    config["data"]["prediction_samples"] = 20
    config["data"]["run_residual_analysis"] = False
    config["data"]["plot_trace"] = False
    return config


def select_model(config: dict):
    model_type = config["model_type"]
    if model_type == "model_n_hv":
        return model_n_hv
    if model_type == "model_n":
        return model_n
    if model_type == "model_empirical":
        return model_empirical
    if model_type == "model_simple":
        return model_simple
    raise NotImplementedError(f"Unsupported model_type for pilot: {model_type}")


def main() -> None:
    config = pilot_config()
    command = [sys.executable, "scripts/run_phase5_pilot.py"]
    bundle = prepare_run_bundle(config, "experimental", command)

    try:
        data_dict = load_all_data(config)
        rng_key = random.PRNGKey(config.get("seed", 0))
        model = select_model(config)
        mcmc = run_inference(model, rng_key, data_dict, config)
        save_results(mcmc, bundle)
    except Exception as exc:
        mark_run_failed(bundle, str(exc))
        raise

    figures_root = REPO_ROOT / "figures" / "tmp"
    before = {path.resolve() for path in figures_root.glob("analysis_*")} if figures_root.exists() else set()
    subprocess.run(
        [sys.executable, "analyze.py", "--results", str(bundle.bundle_dir), "--experimental"],
        cwd=REPO_ROOT,
        check=True,
    )
    after = {path.resolve() for path in figures_root.glob("analysis_*")}
    new_dirs = sorted(after - before)
    if not new_dirs:
        raise RuntimeError("Pilot analysis completed but no new analysis output directory was detected")

    analysis_dir = new_dirs[-1]
    exports_dir = analysis_dir / "exports"
    prediction_path = exports_dir / "prediction_plot_data.csv"
    acceptance_summary_path = exports_dir / "acceptance_summary.json"

    registry_path = REPO_ROOT / "registry" / "paper_evidence_registry.json"
    registry = load_registry(registry_path)
    entry = {
        "evidence_id": f"PILOT-{bundle.run_id}",
        "manuscript_location": "Appendix Pilot end-to-end evidence check",
        "source_run_id": bundle.run_id,
        "source_artifact": str(prediction_path.relative_to(REPO_ROOT)),
        "source_artifact_sha256": sha256_file(prediction_path),
        "status": "candidate",
        "acceptance_review": {
            "summary_artifact": str(acceptance_summary_path.relative_to(REPO_ROOT)),
            "summary_artifact_sha256": sha256_file(acceptance_summary_path),
            "gates_version": 1,
        },
    }
    registry.setdefault("entries", []).append(entry)
    save_registry(registry, registry_path)
    validate_registry_file(registry_path, repo_root=REPO_ROOT)

    print(f"Pilot run bundle: {bundle.bundle_dir}")
    print(f"Pilot analysis output: {analysis_dir}")
    print(f"Pilot registry entry: {entry['evidence_id']}")


if __name__ == "__main__":
    main()

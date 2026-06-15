from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.io.evidence_registry import (
    RegistryValidationError,
    load_registry,
    save_registry,
    validate_registry_document,
    validate_registry_file,
)
from src.io.run_bundle import sha256_file


class EvidenceRegistryTests(unittest.TestCase):
    def _make_artifact(self, root: Path, relative_path: str = "figures/final/analysis_001/exports/prediction_plot_data.csv") -> tuple[Path, str]:
        artifact_path = root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("load,mean_posterior\n0.0,0.1\n")
        return artifact_path, sha256_file(artifact_path)

    def _valid_document(self, root: Path) -> dict:
        artifact_path, checksum = self._make_artifact(root)
        return {
            "schema_version": 1,
            "entries": [
                {
                    "evidence_id": "EVID-001",
                    "manuscript_location": "Figure 7",
                    "source_run_id": "model_empirical_20260615T120000000000Z",
                    "source_artifact": str(artifact_path.relative_to(root)),
                    "source_artifact_sha256": checksum,
                    "status": "candidate",
                }
            ],
        }

    def test_validate_registry_document_accepts_valid_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = validate_registry_document(self._valid_document(root), repo_root=root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["evidence_id"], "EVID-001")

    def test_rejects_missing_source_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            del document["entries"][0]["source_run_id"]

            with self.assertRaisesRegex(RegistryValidationError, "source_run_id"):
                validate_registry_document(document, repo_root=root)

    def test_rejects_missing_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            document["entries"][0]["source_artifact"] = "figures/final/missing.csv"

            with self.assertRaisesRegex(RegistryValidationError, "does not exist"):
                validate_registry_document(document, repo_root=root)

    def test_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            document["entries"][0]["source_artifact_sha256"] = "0" * 64

            with self.assertRaisesRegex(RegistryValidationError, "checksum mismatch"):
                validate_registry_document(document, repo_root=root)

    def test_rejects_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            document["entries"][0]["status"] = "draft"

            with self.assertRaisesRegex(RegistryValidationError, "status must be one of"):
                validate_registry_document(document, repo_root=root)

    def test_rejects_ambiguous_manuscript_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            document["entries"][0]["manuscript_location"] = "results section"

            with self.assertRaisesRegex(RegistryValidationError, "manuscript_location"):
                validate_registry_document(document, repo_root=root)

    def test_rejects_duplicate_evidence_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._valid_document(root)
            document["entries"].append(dict(document["entries"][0]))

            with self.assertRaisesRegex(RegistryValidationError, "Duplicate evidence_id"):
                validate_registry_document(document, repo_root=root)

    def test_load_save_and_validate_registry_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_path = root / "registry" / "paper_evidence_registry.json"
            document = self._valid_document(root)
            save_registry(document, registry_path)

            loaded = load_registry(registry_path)
            self.assertEqual(loaded["schema_version"], 1)

            validated = validate_registry_file(registry_path, repo_root=root)
            self.assertEqual(len(validated), 1)


if __name__ == "__main__":
    unittest.main()

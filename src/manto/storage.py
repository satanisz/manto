"""Immutable, portable analytical results with content integrity checks."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from manto.domain import AnalysisResult, Dataset


def _atomic_text(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ResultStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _folder(self, experiment_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", experiment_id):
            raise ValueError("Invalid experiment identifier")
        return self.directory / experiment_id

    def save(self, result: AnalysisResult, dataset: Dataset | None = None) -> str:
        folder = self._folder(result.experiment_id)
        folder.mkdir(exist_ok=True)
        content = result.model_dump_json(indent=2)
        destination = folder / "result.json"
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise ValueError("An experiment is immutable. Create a new experiment to change it.")
        if not destination.exists():
            _atomic_text(destination, content)
        if not (folder / "manifest.json").exists():
            snapshot_path = None
            if dataset is not None:
                from manto.data import snapshot_dataset

                snapshot_path = str(
                    Path(snapshot_dataset(dataset, folder / "data")).relative_to(folder)
                )
            dependencies = {}
            for name in ("manto", "numpy", "pandas", "scipy", "statsmodels", "langgraph"):
                try:
                    dependencies[name] = version(name)
                except PackageNotFoundError:
                    dependencies[name] = "unavailable"
            manifest = {
                "schema_version": 1,
                "experiment_id": result.experiment_id,
                "saved_at": datetime.now(UTC).isoformat(),
                "result_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "dataset_snapshot": snapshot_path,
                "policy_version": result.policy_version,
                "dependencies": dependencies,
                "target_id": result.request.target_id,
                "dataset_name": result.dataset_name,
                "status": result.status,
                "provenance": result.provenance,
            }
            from importlib.resources import files

            policy_text = files("manto").joinpath("demo_policy.toml").read_text(encoding="utf-8")
            manifest["policy_sha256"] = hashlib.sha256(policy_text.encode()).hexdigest()
            try:
                code_directory = Path(__file__).resolve().parent
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=code_directory,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                dirty = subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=code_directory,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                manifest["code_revision"] = revision.stdout.strip()
                manifest["working_tree_dirty"] = bool(dirty.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                manifest["code_revision"] = "unavailable"
            _atomic_text(folder / "manifest.json", json.dumps(manifest, indent=2))
        return str(destination)

    def load(self, experiment_id: str) -> AnalysisResult:
        folder = self._folder(experiment_id)
        content = (folder / "result.json").read_text(encoding="utf-8")
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(content.encode()).hexdigest() != manifest["result_sha256"]:
            raise ValueError("Saved result failed its integrity check")
        result = AnalysisResult.model_validate_json(content)
        if result.experiment_id != experiment_id:
            raise ValueError("Saved result identifier mismatch")
        return result

    def list_results(self) -> list[dict]:
        results = []
        for manifest in self.directory.glob("*/manifest.json"):
            try:
                item = json.loads(manifest.read_text(encoding="utf-8"))
                if item.get("experiment_id") == manifest.parent.name:
                    results.append(item)
            except (OSError, ValueError):
                continue
        return sorted(results, key=lambda item: item.get("saved_at", ""), reverse=True)

    def load_dataset(self, experiment_id: str) -> Dataset:
        from manto.data import load_snapshot

        self.load(experiment_id)
        folder = self._folder(experiment_id)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        relative = manifest.get("dataset_snapshot")
        if not relative:
            raise ValueError("This result has no saved input snapshot")
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Invalid snapshot reference")
        return load_snapshot(path)

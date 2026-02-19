"""
Artifact Store

Persistent storage for run artifacts (PDFs, extracted data, reports).
Uses filesystem with structured directories per run.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cdr.config import get_settings
from cdr.core.exceptions import StorageError


class ArtifactStore:
    """
    Filesystem-based artifact storage.

    Structure:
        {base_path}/
            {run_id}/
                metadata.json
                pdfs/
                    {record_id}.pdf
                parsed/
                    {record_id}.json
                study_cards/
                    {record_id}.json
                rob2/
                    {record_id}.json
                snippets/
                    {record_id}.json
                reports/
                    report.md
                    report.json
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """
        Initialize artifact store.

        Args:
            base_path: Base directory for artifacts. Defaults to config.
        """
        self._base_path = base_path or get_settings().storage.artifact_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        """Get base path."""
        return self._base_path

    def _run_path(self, run_id: str) -> Path:
        """Get path for a specific run."""
        return self._base_path / run_id

    def _ensure_run_dir(self, run_id: str) -> Path:
        """Ensure run directory exists."""
        run_path = self._run_path(run_id)
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def _subdir(self, run_id: str, subdir: str) -> Path:
        """Get or create a subdirectory within a run."""
        path = self._run_path(run_id) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==================== Run Management ====================

    def init_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> Path:
        """
        Initialize a new run directory.

        Args:
            run_id: Unique run identifier.
            metadata: Optional metadata to store.

        Returns:
            Path to run directory.
        """
        run_path = self._ensure_run_dir(run_id)

        # Create subdirectories
        for subdir in ["pdfs", "parsed", "study_cards", "rob2", "snippets", "reports"]:
            (run_path / subdir).mkdir(exist_ok=True)

        # Write initial metadata
        meta = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized",
            **(metadata or {}),
        }
        self._write_json(run_path / "metadata.json", meta)

        return run_path

    def run_exists(self, run_id: str) -> bool:
        """Check if run directory exists."""
        return self._run_path(run_id).exists()

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its artifacts."""
        run_path = self._run_path(run_id)
        if run_path.exists():
            shutil.rmtree(run_path)

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        return [
            p.name
            for p in self._base_path.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        ]

    def get_run_metadata(self, run_id: str) -> dict[str, Any]:
        """Get run metadata."""
        meta_path = self._run_path(run_id) / "metadata.json"
        if not meta_path.exists():
            raise StorageError(f"Run {run_id} not found")
        return self._read_json(meta_path)

    def update_run_metadata(self, run_id: str, updates: dict[str, Any]) -> None:
        """Update run metadata."""
        meta_path = self._run_path(run_id) / "metadata.json"
        if not meta_path.exists():
            raise StorageError(f"Run {run_id} not found")
        meta = self._read_json(meta_path)
        meta.update(updates)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(meta_path, meta)

    # ==================== PDF Storage ====================

    def store_pdf(self, run_id: str, record_id: str, content: bytes) -> Path:
        """
        Store a PDF file.

        Args:
            run_id: Run identifier.
            record_id: Record/document identifier.
            content: PDF bytes.

        Returns:
            Path to stored file.
        """
        pdf_dir = self._subdir(run_id, "pdfs")
        pdf_path = pdf_dir / f"{record_id}.pdf"
        pdf_path.write_bytes(content)
        return pdf_path

    def get_pdf_path(self, run_id: str, record_id: str) -> Path | None:
        """Get path to stored PDF if it exists."""
        pdf_path = self._run_path(run_id) / "pdfs" / f"{record_id}.pdf"
        return pdf_path if pdf_path.exists() else None

    def pdf_exists(self, run_id: str, record_id: str) -> bool:
        """Check if PDF exists for record."""
        return self.get_pdf_path(run_id, record_id) is not None

    # ==================== JSON Artifact Storage ====================

    def store_parsed(self, run_id: str, record_id: str, data: dict[str, Any]) -> Path:
        """Store parsed document data."""
        return self._store_json_artifact(run_id, "parsed", record_id, data)

    def get_parsed(self, run_id: str, record_id: str) -> dict[str, Any] | None:
        """Get parsed document data."""
        return self._get_json_artifact(run_id, "parsed", record_id)

    def store_study_card(self, run_id: str, record_id: str, data: dict[str, Any]) -> Path:
        """Store StudyCard data."""
        return self._store_json_artifact(run_id, "study_cards", record_id, data)

    def get_study_card(self, run_id: str, record_id: str) -> dict[str, Any] | None:
        """Get StudyCard data."""
        return self._get_json_artifact(run_id, "study_cards", record_id)

    def list_study_cards(self, run_id: str) -> list[str]:
        """List all record IDs with study cards."""
        return self._list_json_artifacts(run_id, "study_cards")

    def store_rob2(self, run_id: str, record_id: str, data: dict[str, Any]) -> Path:
        """Store RoB2 assessment data."""
        return self._store_json_artifact(run_id, "rob2", record_id, data)

    def get_rob2(self, run_id: str, record_id: str) -> dict[str, Any] | None:
        """Get RoB2 assessment data."""
        return self._get_json_artifact(run_id, "rob2", record_id)

    def store_snippets(self, run_id: str, record_id: str, snippets: list[dict[str, Any]]) -> Path:
        """Store extracted snippets for a record."""
        return self._store_json_artifact(run_id, "snippets", record_id, {"snippets": snippets})

    def get_snippets(self, run_id: str, record_id: str) -> list[dict[str, Any]]:
        """Get snippets for a record."""
        data = self._get_json_artifact(run_id, "snippets", record_id)
        return data.get("snippets", []) if data else []

    # ==================== Report Storage ====================

    def store_report(self, run_id: str, content: str, format: str = "md") -> Path:
        """
        Store final report.

        Args:
            run_id: Run identifier.
            content: Report content.
            format: File format (md, json, html).

        Returns:
            Path to stored report.
        """
        report_dir = self._subdir(run_id, "reports")
        report_path = report_dir / f"report.{format}"
        report_path.write_text(content, encoding="utf-8")
        return report_path

    def get_report(self, run_id: str, format: str = "md") -> str | None:
        """Get report content."""
        report_path = self._run_path(run_id) / "reports" / f"report.{format}"
        if report_path.exists():
            return report_path.read_text(encoding="utf-8")
        return None

    # ==================== Generic Artifact Storage ====================

    def store_artifact(
        self, run_id: str, category: str, name: str, data: bytes | str | dict[str, Any]
    ) -> Path:
        """
        Store arbitrary artifact.

        Args:
            run_id: Run identifier.
            category: Artifact category (creates subdirectory).
            name: Artifact name (with extension).
            data: Content to store.

        Returns:
            Path to stored artifact.
        """
        artifact_dir = self._subdir(run_id, category)
        artifact_path = artifact_dir / name

        if isinstance(data, bytes):
            artifact_path.write_bytes(data)
        elif isinstance(data, str):
            artifact_path.write_text(data, encoding="utf-8")
        elif isinstance(data, dict):
            self._write_json(artifact_path, data)
        else:
            raise StorageError(f"Unsupported data type: {type(data)}")

        return artifact_path

    def get_artifact(self, run_id: str, category: str, name: str) -> bytes | None:
        """Get artifact bytes."""
        artifact_path = self._run_path(run_id) / category / name
        if artifact_path.exists():
            return artifact_path.read_bytes()
        return None

    # ==================== Internal Helpers ====================

    def _store_json_artifact(
        self, run_id: str, subdir: str, record_id: str, data: dict[str, Any]
    ) -> Path:
        """Store JSON artifact."""
        artifact_dir = self._subdir(run_id, subdir)
        artifact_path = artifact_dir / f"{record_id}.json"
        self._write_json(artifact_path, data)
        return artifact_path

    def _get_json_artifact(self, run_id: str, subdir: str, record_id: str) -> dict[str, Any] | None:
        """Get JSON artifact."""
        artifact_path = self._run_path(run_id) / subdir / f"{record_id}.json"
        if artifact_path.exists():
            return self._read_json(artifact_path)
        return None

    def _list_json_artifacts(self, run_id: str, subdir: str) -> list[str]:
        """List JSON artifact IDs in a subdirectory."""
        subdir_path = self._run_path(run_id) / subdir
        if not subdir_path.exists():
            return []
        return [p.stem for p in subdir_path.glob("*.json")]

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON with consistent formatting."""
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read JSON file."""
        return json.loads(path.read_text(encoding="utf-8"))

    # ==================== Utilities ====================

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def get_run_size(self, run_id: str) -> int:
        """Get total size of run artifacts in bytes."""
        run_path = self._run_path(run_id)
        if not run_path.exists():
            return 0
        return sum(f.stat().st_size for f in run_path.rglob("*") if f.is_file())

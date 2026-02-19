"""
Run Store

SQLite-based persistence for run state and metadata.
Provides ACID guarantees for workflow state management.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from cdr.config import get_settings
from cdr.core.enums import RunStatus
from cdr.core.exceptions import StorageError


class RunStore:
    """
    SQLite-based run state persistence.

    Tables:
        - runs: Run metadata and status
        - records: Evidence records per run
        - screening_decisions: Inclusion/exclusion decisions
        - checkpoints: LangGraph state checkpoints
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """
        Initialize run store.

        Args:
            db_path: Path to SQLite database. Defaults to config.
        """
        self._db_path = db_path or get_settings().storage.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        """Get database path."""
        return self._db_path

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connection."""
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")  # 30s wait on lock
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise StorageError(f"Database error: {e}") from e
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    pico_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    current_node TEXT,
                    iteration INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    metadata_json TEXT
                );
                
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    external_id TEXT,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    authors_json TEXT,
                    year INTEGER,
                    journal TEXT,
                    doi TEXT,
                    pmid TEXT,
                    study_type TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE (run_id, record_id)
                );
                
                CREATE TABLE IF NOT EXISTS screening_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    included INTEGER NOT NULL,
                    exclusion_reason TEXT,
                    exclusion_rationale TEXT,
                    confidence REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE (run_id, record_id)
                );
                
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );
                
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    overall_score REAL,
                    grade TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
                    UNIQUE (run_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_records_run_id ON records(run_id);
                CREATE INDEX IF NOT EXISTS idx_screening_run_id ON screening_decisions(run_id);
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON evaluations(run_id);
            """)

    # ==================== Run Management ====================

    def create_run(
        self, run_id: str, pico: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Create a new run.

        Args:
            run_id: Unique run identifier.
            pico: PICO question as dict.
            metadata: Optional additional metadata.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, pico_json, status, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    json.dumps(pico),
                    RunStatus.PENDING.value,
                    now,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get run data."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

            if row is None:
                return None

            return self._row_to_dict(row)

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        current_node: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update run status."""
        now = datetime.now(timezone.utc).isoformat()
        completed_at = now if status in (RunStatus.COMPLETED, RunStatus.FAILED) else None

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE runs 
                SET status = ?, current_node = ?, updated_at = ?, 
                    completed_at = COALESCE(?, completed_at),
                    error_message = COALESCE(?, error_message)
                WHERE run_id = ?
                """,
                (status.value, current_node, now, completed_at, error_message, run_id),
            )

    def increment_iteration(self, run_id: str) -> int:
        """Increment and return the iteration counter."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE runs SET iteration = iteration + 1, updated_at = ? WHERE run_id = ?",
                (datetime.now(timezone.utc).isoformat(), run_id),
            )
            row = conn.execute("SELECT iteration FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return row["iteration"] if row else 0

    def list_runs(self, status: RunStatus | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List runs with optional status filter."""
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()

            return [self._row_to_dict(row) for row in rows]

    def delete_run(self, run_id: str) -> None:
        """Delete run and all associated data."""
        with self._connection() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    # ==================== Record Management ====================

    def add_record(self, run_id: str, record: dict[str, Any]) -> None:
        """Add a record to a run.

        Supports both 'id' (legacy) and 'record_id' (current schema) field names.
        """
        now = datetime.now(timezone.utc).isoformat()
        # Support both 'id' (legacy) and 'record_id' (current schema)
        record_id = record.get("record_id") or record.get("id")
        if not record_id:
            raise ValueError("Record must have 'record_id' or 'id' field")

        # Handle RecordSource enum if present
        source = record.get("source")
        if hasattr(source, "value"):
            source = source.value

        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO records 
                (run_id, record_id, source, external_id, title, abstract, 
                 authors_json, year, journal, doi, pmid, study_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    record_id,
                    source,
                    record.get("external_id"),
                    record["title"],
                    record.get("abstract"),
                    json.dumps(record.get("authors", [])),
                    record.get("year"),
                    record.get("journal"),
                    record.get("doi"),
                    record.get("pmid"),
                    record.get("study_type"),
                    json.dumps(record.get("metadata", {})),
                    now,
                ),
            )

    def add_records(self, run_id: str, records: list[dict[str, Any]]) -> None:
        """Add multiple records to a run."""
        for record in records:
            self.add_record(run_id, record)

    def get_records(self, run_id: str) -> list[dict[str, Any]]:
        """Get all records for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()

            return [self._record_row_to_dict(row) for row in rows]

    def get_record(self, run_id: str, record_id: str) -> dict[str, Any] | None:
        """Get a specific record."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM records WHERE run_id = ? AND record_id = ?", (run_id, record_id)
            ).fetchone()

            return self._record_row_to_dict(row) if row else None

    def count_records(self, run_id: str) -> int:
        """Count records in a run."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM records WHERE run_id = ?", (run_id,)
            ).fetchone()
            return row["count"] if row else 0

    # ==================== Screening Decisions ====================

    def add_screening_decision(
        self,
        run_id: str,
        record_id: str,
        included: bool,
        exclusion_reason: str | None = None,
        exclusion_rationale: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Add a screening decision."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO screening_decisions
                (run_id, record_id, included, exclusion_reason, exclusion_rationale, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    record_id,
                    int(included),
                    exclusion_reason,
                    exclusion_rationale,
                    confidence,
                    now,
                ),
            )

    def get_screening_decisions(self, run_id: str) -> list[dict[str, Any]]:
        """Get all screening decisions for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM screening_decisions WHERE run_id = ?", (run_id,)
            ).fetchall()

            return [
                {
                    "record_id": row["record_id"],
                    "included": bool(row["included"]),
                    "exclusion_reason": row["exclusion_reason"],
                    "exclusion_rationale": row["exclusion_rationale"],
                    "confidence": row["confidence"],
                }
                for row in rows
            ]

    def get_included_record_ids(self, run_id: str) -> list[str]:
        """Get IDs of included records."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT record_id FROM screening_decisions WHERE run_id = ? AND included = 1",
                (run_id,),
            ).fetchall()
            return [row["record_id"] for row in rows]

    def count_included(self, run_id: str) -> int:
        """Count included records."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM screening_decisions WHERE run_id = ? AND included = 1",
                (run_id,),
            ).fetchone()
            return row["count"] if row else 0

    # ==================== Checkpoints ====================

    def save_checkpoint(self, run_id: str, node: str, state: dict[str, Any]) -> None:
        """Save a workflow checkpoint."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (run_id, node, state_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, node, json.dumps(state, default=str), now),
            )

    def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Get the latest checkpoint for a run."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints 
                WHERE run_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "run_id": row["run_id"],
                "node": row["node"],
                "state": json.loads(row["state_json"]),
                "created_at": row["created_at"],
            }

    def get_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """Get all checkpoints for a run."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()

            return [
                {
                    "run_id": row["run_id"],
                    "node": row["node"],
                    "state": json.loads(row["state_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # ==================== Helpers ====================

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert run row to dict."""
        return {
            "run_id": row["run_id"],
            "pico": json.loads(row["pico_json"]),
            "status": row["status"],
            "current_node": row["current_node"],
            "iteration": row["iteration"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        }

    def _record_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert record row to dict."""
        return {
            "id": row["record_id"],
            "source": row["source"],
            "external_id": row["external_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "authors": json.loads(row["authors_json"]) if row["authors_json"] else [],
            "year": row["year"],
            "journal": row["journal"],
            "doi": row["doi"],
            "pmid": row["pmid"],
            "study_type": row["study_type"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
        }

    # ==================== Evaluation Management ====================

    def save_evaluation(
        self,
        run_id: str,
        report: dict[str, Any],
        overall_score: float | None = None,
        grade: str | None = None,
    ) -> None:
        """
        Save evaluation report for a run.

        Args:
            run_id: Run identifier.
            report: Full evaluation report as dict.
            overall_score: Optional overall score (0-1).
            grade: Optional letter grade (A-F).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations 
                (run_id, report_json, overall_score, grade, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, json.dumps(report), overall_score, grade, now),
            )

    def get_evaluation(self, run_id: str) -> dict[str, Any] | None:
        """
        Get evaluation report for a run.

        Args:
            run_id: Run identifier.

        Returns:
            Evaluation data dict or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE run_id = ?", (run_id,)).fetchone()

            if row is None:
                return None

            return {
                "run_id": row["run_id"],
                "report": json.loads(row["report_json"]),
                "overall_score": row["overall_score"],
                "grade": row["grade"],
                "created_at": row["created_at"],
            }

    def delete_evaluation(self, run_id: str) -> bool:
        """
        Delete evaluation for a run.

        Args:
            run_id: Run identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM evaluations WHERE run_id = ?", (run_id,))
            return cursor.rowcount > 0

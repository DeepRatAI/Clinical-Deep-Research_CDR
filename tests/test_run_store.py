"""
Tests for RunStore persistence module.

MEDIUM-6 verification: SQLite/JSON storage for run state.
Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-6

Tests:
1. Run lifecycle management (create, get, update, list, delete)
2. Record management within runs
3. Screening decision persistence
4. Checkpoint save/restore
5. Error handling and edge cases
"""

import json
import tempfile
from pathlib import Path

import pytest

from cdr.core.enums import RunStatus
from cdr.storage.run_store import RunStore


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_runs.db"
        yield db_path


@pytest.fixture
def store(temp_db):
    """Create RunStore with temporary database."""
    return RunStore(db_path=temp_db)


class TestRunStoreInit:
    """Tests for RunStore initialization."""

    def test_creates_database_file(self, temp_db):
        """Database file is created on init."""
        store = RunStore(db_path=temp_db)
        assert temp_db.exists()

    def test_creates_parent_directories(self):
        """Parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "deep" / "test.db"
            store = RunStore(db_path=db_path)
            assert db_path.exists()

    def test_schema_is_initialized(self, store):
        """Tables are created on init."""
        # Verify by trying to insert/query
        store.create_run("test-run", {"population": "test"})
        run = store.get_run("test-run")
        assert run is not None


class TestRunManagement:
    """Tests for run lifecycle management."""

    def test_create_run(self, store):
        """Create run with PICO data."""
        pico = {
            "population": "Adults with diabetes",
            "intervention": "Metformin",
            "comparator": "Placebo",
            "outcome": "HbA1c reduction",
        }
        store.create_run("run-001", pico)

        run = store.get_run("run-001")
        assert run is not None
        assert run["run_id"] == "run-001"
        assert run["pico"] == pico
        assert run["status"] == "pending"

    def test_create_run_with_metadata(self, store):
        """Create run with additional metadata."""
        pico = {"population": "test"}
        metadata = {"source": "api", "user_id": "user-123"}
        store.create_run("run-meta", pico, metadata=metadata)

        run = store.get_run("run-meta")
        assert run["metadata"] == metadata

    def test_get_nonexistent_run_returns_none(self, store):
        """Getting non-existent run returns None."""
        run = store.get_run("nonexistent")
        assert run is None

    def test_update_run_status(self, store):
        """Update run status."""
        store.create_run("run-status", {"p": "test"})

        store.update_run_status("run-status", RunStatus.RUNNING, current_node="retrieve")
        run = store.get_run("run-status")
        assert run["status"] == "running"
        assert run["current_node"] == "retrieve"

    def test_update_run_status_completed(self, store):
        """Completed status sets completed_at."""
        store.create_run("run-done", {"p": "test"})

        store.update_run_status("run-done", RunStatus.COMPLETED)
        run = store.get_run("run-done")
        assert run["status"] == "completed"
        assert run["completed_at"] is not None

    def test_update_run_status_failed_with_error(self, store):
        """Failed status can include error message."""
        store.create_run("run-fail", {"p": "test"})

        store.update_run_status("run-fail", RunStatus.FAILED, error_message="API timeout")
        run = store.get_run("run-fail")
        assert run["status"] == "failed"
        assert run["error_message"] == "API timeout"

    def test_increment_iteration(self, store):
        """Increment iteration counter."""
        store.create_run("run-iter", {"p": "test"})

        iter1 = store.increment_iteration("run-iter")
        iter2 = store.increment_iteration("run-iter")
        iter3 = store.increment_iteration("run-iter")

        assert iter1 == 1
        assert iter2 == 2
        assert iter3 == 3

    def test_list_runs(self, store):
        """List all runs."""
        store.create_run("run-a", {"p": "a"})
        store.create_run("run-b", {"p": "b"})
        store.create_run("run-c", {"p": "c"})

        runs = store.list_runs()
        assert len(runs) == 3
        # Most recent first
        run_ids = [r["run_id"] for r in runs]
        assert set(run_ids) == {"run-a", "run-b", "run-c"}

    def test_list_runs_with_status_filter(self, store):
        """List runs filtered by status."""
        store.create_run("run-1", {"p": "1"})
        store.create_run("run-2", {"p": "2"})
        store.create_run("run-3", {"p": "3"})

        store.update_run_status("run-1", RunStatus.COMPLETED)
        store.update_run_status("run-2", RunStatus.RUNNING)

        completed = store.list_runs(status=RunStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0]["run_id"] == "run-1"

        pending = store.list_runs(status=RunStatus.PENDING)
        assert len(pending) == 1
        assert pending[0]["run_id"] == "run-3"

    def test_delete_run(self, store):
        """Delete run removes it."""
        store.create_run("run-delete", {"p": "test"})
        assert store.get_run("run-delete") is not None

        store.delete_run("run-delete")
        assert store.get_run("run-delete") is None


class TestRecordManagement:
    """Tests for record management within runs."""

    @pytest.fixture
    def store_with_run(self, store):
        """Store with an existing run."""
        store.create_run("test-run", {"p": "test"})
        return store

    def test_add_record(self, store_with_run):
        """Add a record to a run."""
        record = {
            "id": "rec-001",
            "source": "pubmed",
            "title": "Test Study Title",
            "abstract": "Study abstract text...",
            "pmid": "12345678",
            "doi": "10.1234/test",
            "year": 2024,
        }
        store_with_run.add_record("test-run", record)

        records = store_with_run.get_records("test-run")
        assert len(records) == 1
        assert records[0]["id"] == "rec-001"
        assert records[0]["title"] == "Test Study Title"
        assert records[0]["pmid"] == "12345678"

    def test_add_records_batch(self, store_with_run):
        """Add multiple records at once."""
        records = [
            {"id": "rec-1", "source": "pubmed", "title": "Study 1"},
            {"id": "rec-2", "source": "pubmed", "title": "Study 2"},
            {"id": "rec-3", "source": "cochrane", "title": "Study 3"},
        ]
        store_with_run.add_records("test-run", records)

        result = store_with_run.get_records("test-run")
        assert len(result) == 3

    def test_get_specific_record(self, store_with_run):
        """Get a specific record by ID."""
        store_with_run.add_record("test-run", {"id": "rec-x", "source": "pubmed", "title": "X"})
        store_with_run.add_record("test-run", {"id": "rec-y", "source": "pubmed", "title": "Y"})

        record = store_with_run.get_record("test-run", "rec-x")
        assert record is not None
        assert record["id"] == "rec-x"

    def test_count_records(self, store_with_run):
        """Count records in a run."""
        store_with_run.add_record("test-run", {"id": "1", "source": "s", "title": "T"})
        store_with_run.add_record("test-run", {"id": "2", "source": "s", "title": "T"})

        assert store_with_run.count_records("test-run") == 2

    def test_add_record_with_authors(self, store_with_run):
        """Records can include author list."""
        record = {
            "id": "rec-auth",
            "source": "pubmed",
            "title": "Multi-author Study",
            "authors": ["Smith J", "Jones K", "Wang L"],
        }
        store_with_run.add_record("test-run", record)

        result = store_with_run.get_record("test-run", "rec-auth")
        assert result["authors"] == ["Smith J", "Jones K", "Wang L"]


class TestScreeningDecisions:
    """Tests for screening decision persistence."""

    @pytest.fixture
    def store_with_records(self, store):
        """Store with run and records."""
        store.create_run("screen-run", {"p": "test"})
        store.add_records(
            "screen-run",
            [
                {"id": "rec-1", "source": "pubmed", "title": "Study 1"},
                {"id": "rec-2", "source": "pubmed", "title": "Study 2"},
                {"id": "rec-3", "source": "pubmed", "title": "Study 3"},
            ],
        )
        return store

    def test_add_inclusion_decision(self, store_with_records):
        """Add inclusion decision."""
        store_with_records.add_screening_decision(
            "screen-run", "rec-1", included=True, confidence=0.95
        )

        decisions = store_with_records.get_screening_decisions("screen-run")
        assert len(decisions) == 1
        assert decisions[0]["record_id"] == "rec-1"
        assert decisions[0]["included"] is True
        assert decisions[0]["confidence"] == 0.95

    def test_add_exclusion_decision(self, store_with_records):
        """Add exclusion decision with reason."""
        store_with_records.add_screening_decision(
            "screen-run",
            "rec-2",
            included=False,
            exclusion_reason="wrong_population",
            exclusion_rationale="Study included pediatric patients only",
            confidence=0.88,
        )

        decisions = store_with_records.get_screening_decisions("screen-run")
        decision = next(d for d in decisions if d["record_id"] == "rec-2")
        assert decision["included"] is False
        assert decision["exclusion_reason"] == "wrong_population"

    def test_get_included_record_ids(self, store_with_records):
        """Get IDs of included records."""
        store_with_records.add_screening_decision("screen-run", "rec-1", included=True)
        store_with_records.add_screening_decision("screen-run", "rec-2", included=False)
        store_with_records.add_screening_decision("screen-run", "rec-3", included=True)

        included = store_with_records.get_included_record_ids("screen-run")
        assert set(included) == {"rec-1", "rec-3"}

    def test_count_included(self, store_with_records):
        """Count included records."""
        store_with_records.add_screening_decision("screen-run", "rec-1", included=True)
        store_with_records.add_screening_decision("screen-run", "rec-2", included=False)
        store_with_records.add_screening_decision("screen-run", "rec-3", included=True)

        assert store_with_records.count_included("screen-run") == 2


class TestCheckpoints:
    """Tests for workflow checkpoint persistence."""

    @pytest.fixture
    def store_with_run(self, store):
        """Store with an existing run."""
        store.create_run("checkpoint-run", {"p": "test"})
        return store

    def test_save_checkpoint(self, store_with_run):
        """Save a workflow checkpoint."""
        state = {
            "run_id": "checkpoint-run",
            "question": "Test question",
            "records": [{"id": "1", "title": "Study"}],
            "status": "running",
        }
        store_with_run.save_checkpoint("checkpoint-run", "retrieve", state)

        checkpoints = store_with_run.get_checkpoints("checkpoint-run")
        assert len(checkpoints) == 1
        assert checkpoints[0]["node"] == "retrieve"
        assert checkpoints[0]["state"]["question"] == "Test question"

    def test_save_multiple_checkpoints(self, store_with_run):
        """Save checkpoints at different nodes."""
        store_with_run.save_checkpoint("checkpoint-run", "retrieve", {"step": 1})
        store_with_run.save_checkpoint("checkpoint-run", "screen", {"step": 2})
        store_with_run.save_checkpoint("checkpoint-run", "synthesize", {"step": 3})

        checkpoints = store_with_run.get_checkpoints("checkpoint-run")
        assert len(checkpoints) == 3
        nodes = [c["node"] for c in checkpoints]
        assert nodes == ["retrieve", "screen", "synthesize"]

    def test_get_latest_checkpoint(self, store_with_run):
        """Get most recent checkpoint."""
        store_with_run.save_checkpoint("checkpoint-run", "retrieve", {"v": 1})
        store_with_run.save_checkpoint("checkpoint-run", "screen", {"v": 2})
        store_with_run.save_checkpoint("checkpoint-run", "synthesize", {"v": 3})

        latest = store_with_run.get_latest_checkpoint("checkpoint-run")
        assert latest is not None
        assert latest["node"] == "synthesize"
        assert latest["state"]["v"] == 3

    def test_get_latest_checkpoint_nonexistent_run(self, store):
        """Latest checkpoint for nonexistent run returns None."""
        result = store.get_latest_checkpoint("nonexistent-run")
        assert result is None


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    def test_delete_run_cascades_to_records(self, store):
        """Deleting run also deletes its records."""
        store.create_run("cascade-run", {"p": "test"})
        store.add_records(
            "cascade-run",
            [
                {"id": "r1", "source": "s", "title": "T"},
                {"id": "r2", "source": "s", "title": "T"},
            ],
        )
        assert store.count_records("cascade-run") == 2

        store.delete_run("cascade-run")
        # After delete, records should be gone too
        assert store.count_records("cascade-run") == 0

    def test_delete_run_cascades_to_screening(self, store):
        """Deleting run also deletes screening decisions."""
        store.create_run("cascade-run", {"p": "test"})
        store.add_record("cascade-run", {"id": "r1", "source": "s", "title": "T"})
        store.add_screening_decision("cascade-run", "r1", included=True)

        assert len(store.get_screening_decisions("cascade-run")) == 1

        store.delete_run("cascade-run")
        assert len(store.get_screening_decisions("cascade-run")) == 0

    def test_delete_run_cascades_to_checkpoints(self, store):
        """Deleting run also deletes checkpoints."""
        store.create_run("cascade-run", {"p": "test"})
        store.save_checkpoint("cascade-run", "node", {"state": "data"})

        assert len(store.get_checkpoints("cascade-run")) == 1

        store.delete_run("cascade-run")
        assert len(store.get_checkpoints("cascade-run")) == 0


class TestConcurrency:
    """
    Tests for concurrent access to RunStore.

    SQLite with WAL mode should handle concurrent reads and serialized writes.
    These tests validate thread safety for multi-run scenarios.

    Refs: CDR_Integral_Audit_2026-01-20.md (session 2026-01-22) - concurrencia pendiente
    """

    def test_concurrent_reads_same_run(self, temp_db):
        """Multiple threads can read the same run concurrently."""
        import threading
        import time

        store = RunStore(db_path=temp_db)
        store.create_run("shared-run", {"population": "test"})
        store.add_records(
            "shared-run",
            [{"id": f"rec-{i}", "source": "pubmed", "title": f"Study {i}"} for i in range(10)],
        )

        results = []
        errors = []

        def reader(thread_id):
            try:
                # Each thread creates its own connection via RunStore
                reader_store = RunStore(db_path=temp_db)
                for _ in range(5):
                    run = reader_store.get_run("shared-run")
                    records = reader_store.get_records("shared-run")
                    results.append((thread_id, run is not None, len(records)))
                    time.sleep(0.01)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent read errors: {errors}"
        assert len(results) == 25  # 5 threads x 5 iterations
        # All reads should succeed
        for thread_id, run_found, record_count in results:
            assert run_found, f"Thread {thread_id} failed to read run"
            assert record_count == 10, f"Thread {thread_id} got wrong record count"

    def test_concurrent_writes_different_runs(self, temp_db):
        """Multiple threads can write to different runs concurrently."""
        import threading
        import time

        errors = []
        run_ids = [f"run-{i}" for i in range(5)]

        # Pre-create runs sequentially (realistic: runs created via API one at a time)
        # This avoids the rare SQLite lock contention on run creation
        setup_store = RunStore(db_path=temp_db)
        for run_id in run_ids:
            setup_store.create_run(run_id, {"population": f"pop-{run_id}"})

        def writer(run_id):
            try:
                writer_store = RunStore(db_path=temp_db)
                # Retry logic for transient lock errors (increased for CI environments)
                max_retries = 5

                for j in range(10):
                    for attempt in range(max_retries):
                        try:
                            writer_store.add_record(
                                run_id,
                                {"id": f"{run_id}-rec-{j}", "source": "pubmed", "title": f"S{j}"},
                            )
                            break
                        except Exception as e:
                            if "locked" in str(e) and attempt < max_retries - 1:
                                time.sleep(0.2 * (attempt + 1))  # Exponential backoff
                                continue
                            raise

                # Also retry update_run_status for transient lock errors
                for attempt in range(max_retries):
                    try:
                        writer_store.update_run_status(run_id, RunStatus.COMPLETED)
                        break
                    except Exception as e:
                        if "locked" in str(e) and attempt < max_retries - 1:
                            time.sleep(0.2 * (attempt + 1))  # Exponential backoff
                            continue
                        raise
            except Exception as e:
                errors.append((run_id, str(e)))

        threads = [threading.Thread(target=writer, args=(run_id,)) for run_id in run_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify all runs were created correctly
        verify_store = RunStore(db_path=temp_db)
        for run_id in run_ids:
            run = verify_store.get_run(run_id)
            assert run is not None, f"Run {run_id} not found"
            assert run["status"] == "completed"
            assert verify_store.count_records(run_id) == 10

    def test_concurrent_writes_same_run(self, temp_db):
        """Multiple threads writing to same run are serialized correctly."""
        import threading

        store = RunStore(db_path=temp_db)
        store.create_run("contested-run", {"population": "test"})

        errors = []

        def add_records(thread_id, count):
            try:
                writer_store = RunStore(db_path=temp_db)
                for i in range(count):
                    writer_store.add_record(
                        "contested-run",
                        {
                            "id": f"t{thread_id}-rec-{i}",
                            "source": "pubmed",
                            "title": f"T{thread_id}S{i}",
                        },
                    )
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 5 threads each adding 5 records = 25 total
        threads = [threading.Thread(target=add_records, args=(i, 5)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write to same run errors: {errors}"

        # All 25 records should be present
        verify_store = RunStore(db_path=temp_db)
        record_count = verify_store.count_records("contested-run")
        assert record_count == 25, f"Expected 25 records, got {record_count}"

    def test_concurrent_read_write(self, temp_db):
        """Readers can access data while writers are active."""
        import threading
        import time

        store = RunStore(db_path=temp_db)
        store.create_run("rw-run", {"population": "test"})

        write_complete = threading.Event()
        read_results = []
        errors = []

        def writer():
            try:
                w_store = RunStore(db_path=temp_db)
                for i in range(20):
                    w_store.add_record(
                        "rw-run", {"id": f"rec-{i}", "source": "pubmed", "title": f"Study {i}"}
                    )
                    time.sleep(0.01)
                write_complete.set()
            except Exception as e:
                errors.append(("writer", str(e)))

        def reader():
            try:
                r_store = RunStore(db_path=temp_db)
                while not write_complete.is_set():
                    records = r_store.get_records("rw-run")
                    read_results.append(len(records))
                    time.sleep(0.02)
                # Final read after writer completes
                final_count = r_store.count_records("rw-run")
                read_results.append(final_count)
            except Exception as e:
                errors.append(("reader", str(e)))

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert not errors, f"Read/write errors: {errors}"
        # Final read should see all 20 records
        assert read_results[-1] == 20, f"Final count should be 20, got {read_results[-1]}"
        # Reads during write should show monotonically increasing counts (or same)
        for i in range(len(read_results) - 1):
            assert read_results[i] <= read_results[i + 1], "Record count decreased during reads"

    def test_status_update_isolation(self, temp_db):
        """Status updates are isolated and don't conflict."""
        import threading

        store = RunStore(db_path=temp_db)
        run_ids = [f"iso-run-{i}" for i in range(10)]

        for run_id in run_ids:
            store.create_run(run_id, {"p": "test"})

        errors = []
        final_statuses = [RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INSUFFICIENT_EVIDENCE]

        def update_status(run_id, idx):
            try:
                u_store = RunStore(db_path=temp_db)
                status = final_statuses[idx % len(final_statuses)]
                u_store.update_run_status(run_id, RunStatus.RUNNING, current_node="processing")
                u_store.update_run_status(run_id, status)
            except Exception as e:
                errors.append((run_id, str(e)))

        threads = [
            threading.Thread(target=update_status, args=(run_id, i))
            for i, run_id in enumerate(run_ids)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Status update errors: {errors}"

        # Verify final statuses
        verify_store = RunStore(db_path=temp_db)
        for i, run_id in enumerate(run_ids):
            run = verify_store.get_run(run_id)
            assert run is not None, f"{run_id} not found"
            expected_status = final_statuses[i % len(final_statuses)].value
            assert run["status"] == expected_status, f"{run_id} has wrong status"


class TestEvaluationManagement:
    """Tests for evaluation persistence."""

    def test_save_evaluation(self, store):
        """Test saving evaluation report."""
        pico = {"population": "Test", "intervention": "Drug", "outcome": "Cure"}
        store.create_run("run_eval_1", pico)

        report = {
            "dimensions": [
                {
                    "name": "evidence_quality",
                    "score": 0.8,
                    "grade": "B",
                    "rationale": "Good evidence base",
                },
                {
                    "name": "methodology",
                    "score": 0.7,
                    "grade": "C",
                    "rationale": "Some limitations",
                },
            ],
            "overall_score": 0.75,
            "overall_grade": "B",
            "strengths": ["Strong RCT evidence", "Well-defined population"],
            "weaknesses": ["Limited follow-up duration"],
            "recommendations": ["Conduct longer trials"],
        }

        store.save_evaluation("run_eval_1", report, overall_score=0.75, grade="B")

        eval_data = store.get_evaluation("run_eval_1")
        assert eval_data is not None
        assert eval_data["overall_score"] == 0.75
        assert eval_data["grade"] == "B"
        assert eval_data["report"]["overall_grade"] == "B"
        assert len(eval_data["report"]["dimensions"]) == 2

    def test_get_nonexistent_evaluation(self, store):
        """Get evaluation for nonexistent run returns None."""
        pico = {"population": "Test", "intervention": "Drug", "outcome": "Cure"}
        store.create_run("run_no_eval", pico)

        eval_data = store.get_evaluation("run_no_eval")
        assert eval_data is None

    def test_update_evaluation(self, store):
        """Test updating evaluation report (replace)."""
        pico = {"population": "Test", "intervention": "Drug", "outcome": "Cure"}
        store.create_run("run_eval_update", pico)

        # Save initial evaluation
        store.save_evaluation("run_eval_update", {"score": 0.5}, overall_score=0.5, grade="C")

        # Update with new evaluation
        store.save_evaluation("run_eval_update", {"score": 0.9}, overall_score=0.9, grade="A")

        eval_data = store.get_evaluation("run_eval_update")
        assert eval_data["overall_score"] == 0.9
        assert eval_data["grade"] == "A"

    def test_delete_evaluation(self, store):
        """Test deleting evaluation."""
        pico = {"population": "Test", "intervention": "Drug", "outcome": "Cure"}
        store.create_run("run_eval_del", pico)
        store.save_evaluation("run_eval_del", {"test": True}, overall_score=0.7, grade="B")

        deleted = store.delete_evaluation("run_eval_del")
        assert deleted is True

        eval_data = store.get_evaluation("run_eval_del")
        assert eval_data is None

    def test_delete_nonexistent_evaluation(self, store):
        """Delete nonexistent evaluation returns False."""
        deleted = store.delete_evaluation("nonexistent_run")
        assert deleted is False

    def test_evaluation_cascade_on_run_delete(self, store):
        """Evaluation is deleted when run is deleted."""
        pico = {"population": "Test", "intervention": "Drug", "outcome": "Cure"}
        store.create_run("run_cascade_eval", pico)
        store.save_evaluation("run_cascade_eval", {"test": True}, overall_score=0.8, grade="B")

        # Verify evaluation exists
        assert store.get_evaluation("run_cascade_eval") is not None

        # Delete run
        store.delete_run("run_cascade_eval")

        # Evaluation should be deleted
        assert store.get_evaluation("run_cascade_eval") is None

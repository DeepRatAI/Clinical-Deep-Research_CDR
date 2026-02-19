"""
Tests for CDRRunner persistence integration.

MEDIUM-6 verification: CDRRunner correctly persists run state to RunStore.
Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-6

Tests:
1. CDRRunner accepts RunStore parameter
2. Run creation is persisted
3. Final state (success) is persisted with records and screening
4. Failed runs are persisted with error message
5. E2E test with mock graph
"""

import asyncio
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cdr.core.enums import ExclusionReason, RecordSource, RunStatus
from cdr.core.schemas import CDRState, Record, ScreeningDecision
from cdr.storage.run_store import RunStore


def _make_record(
    record_id: str, title: str, abstract: str = "Abstract text", pmid: str | None = None
) -> Record:
    """Helper to create a valid Record with required fields."""
    content = f"{title}|{abstract}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return Record(
        record_id=record_id,
        source=RecordSource.PUBMED,
        content_hash=content_hash,
        title=title,
        abstract=abstract,
        pmid=pmid,
    )


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


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="test response")
    return provider


class TestCDRRunnerPersistenceInit:
    """Tests for CDRRunner initialization with RunStore."""

    def test_runner_accepts_run_store_parameter(self, mock_llm_provider, store):
        """CDRRunner accepts run_store parameter."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(
            llm_provider=mock_llm_provider,
            run_store=store,
        )
        assert runner.run_store is store

    def test_runner_works_without_run_store(self, mock_llm_provider):
        """CDRRunner works without run_store (backward compatible)."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=mock_llm_provider)
        assert runner.run_store is None


class TestCDRRunnerPersistenceOnSuccess:
    """Tests for persistence on successful run completion."""

    def test_run_creation_is_persisted(self, mock_llm_provider, store):
        """Run is created in RunStore at start."""
        from cdr.orchestration.graph import CDRRunner

        # Mock the graph to return immediately
        mock_state = CDRState(
            run_id="test-run",
            question="Test question",
            status=RunStatus.COMPLETED,
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Test question", run_id="test-run"))

        # Verify run was created
        run_data = store.get_run("test-run")
        assert run_data is not None
        assert run_data["run_id"] == "test-run"

    def test_final_status_is_persisted(self, mock_llm_provider, store):
        """Final status is updated in RunStore."""
        from cdr.orchestration.graph import CDRRunner

        mock_state = CDRState(
            run_id="status-run",
            question="Test question",
            status=RunStatus.INSUFFICIENT_EVIDENCE,
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Test question", run_id="status-run"))

        run_data = store.get_run("status-run")
        assert run_data["status"] == RunStatus.INSUFFICIENT_EVIDENCE.value

    def test_records_are_persisted(self, mock_llm_provider, store):
        """Records are persisted to RunStore."""
        from cdr.orchestration.graph import CDRRunner

        record1 = _make_record("rec-001", "Test Study 1")
        record2 = _make_record("rec-002", "Test Study 2", "Abstract text 2")

        mock_state = CDRState(
            run_id="rec-run",
            question="Test question",
            status=RunStatus.COMPLETED,
            retrieved_records=[record1, record2],
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Test question", run_id="rec-run"))

        # Verify records were persisted
        records = store.get_records("rec-run")
        assert len(records) == 2
        assert records[0]["id"] == "rec-001"
        assert records[1]["id"] == "rec-002"

    def test_screening_decisions_are_persisted(self, mock_llm_provider, store):
        """Screening decisions are persisted to RunStore."""
        from cdr.orchestration.graph import CDRRunner

        sd1 = ScreeningDecision(record_id="rec-001", included=True)
        sd2 = ScreeningDecision(
            record_id="rec-002",
            included=False,
            reason_code=ExclusionReason.POPULATION_MISMATCH,
            reason_text="Population does not match",
        )

        mock_state = CDRState(
            run_id="screen-run",
            question="Test question",
            status=RunStatus.COMPLETED,
            screened=[sd1, sd2],
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Test question", run_id="screen-run"))

        # Verify screening decisions
        decisions = store.get_screening_decisions("screen-run")
        assert len(decisions) == 2

        included = next(d for d in decisions if d["record_id"] == "rec-001")
        assert included["included"] is True

        excluded = next(d for d in decisions if d["record_id"] == "rec-002")
        assert excluded["included"] is False
        assert excluded["exclusion_reason"] == "population_mismatch"

    def test_checkpoint_is_saved_on_completion(self, mock_llm_provider, store):
        """Checkpoint is saved with final state."""
        from cdr.orchestration.graph import CDRRunner

        mock_state = CDRState(
            run_id="check-run",
            question="Test question",
            status=RunStatus.COMPLETED,
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Test question", run_id="check-run"))

        # Verify checkpoint
        checkpoint = store.get_latest_checkpoint("check-run")
        assert checkpoint is not None
        assert checkpoint["node"] == "completed"
        assert checkpoint["state"]["run_id"] == "check-run"


class TestCDRRunnerPersistenceOnFailure:
    """Tests for persistence on failed runs."""

    def test_failed_run_status_is_persisted(self, mock_llm_provider, store):
        """Failed run status is persisted with error message."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        # Mock graph to raise exception
        with patch.object(
            runner.graph, "ainvoke", new=AsyncMock(side_effect=ValueError("Test error"))
        ):
            asyncio.run(runner.run("Test question", run_id="fail-run"))

        # Verify failed status
        run_data = store.get_run("fail-run")
        assert run_data["status"] == RunStatus.FAILED.value
        assert "Test error" in run_data["error_message"]

    def test_persistence_failure_does_not_break_run(self, mock_llm_provider, temp_db):
        """Persistence failure is logged but does not break the run."""
        from cdr.orchestration.graph import CDRRunner

        # Create a store that will fail on update
        store = RunStore(db_path=temp_db)

        mock_state = CDRState(
            run_id="persist-fail",
            question="Test question",
            status=RunStatus.COMPLETED,
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        # Make update_run_status fail and mock graph
        with (
            patch.object(store, "update_run_status", side_effect=Exception("DB error")),
            patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)),
        ):
            # Should not raise even though persistence fails
            result = asyncio.run(runner.run("Test question", run_id="persist-fail"))

        # Run still completes
        assert result.status == RunStatus.COMPLETED


class TestCDRRunnerPersistenceE2E:
    """End-to-end persistence tests."""

    def test_full_lifecycle_persisted(self, mock_llm_provider, store):
        """Full run lifecycle is persisted correctly."""
        from cdr.orchestration.graph import CDRRunner

        # Create realistic state
        record = _make_record(
            "pmid-12345",
            "Efficacy of Treatment X",
            "A randomized controlled trial...",
            pmid="12345",
        )
        screening = ScreeningDecision(record_id="pmid-12345", included=True)

        mock_state = CDRState(
            run_id="e2e-run",
            question="Does treatment X work for condition Y?",
            status=RunStatus.COMPLETED,
            retrieved_records=[record],
            screened=[screening],
        )

        runner = CDRRunner(
            llm_provider=mock_llm_provider,
            run_store=store,
            dod_level=2,
        )

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(
                runner.run(
                    "Does treatment X work for condition Y?",
                    run_id="e2e-run",
                    max_results=50,
                )
            )

        # Verify complete persistence
        run = store.get_run("e2e-run")
        assert run["status"] == "completed"
        assert run["metadata"]["dod_level"] == 2

        records = store.get_records("e2e-run")
        assert len(records) == 1
        assert records[0]["pmid"] == "12345"

        decisions = store.get_screening_decisions("e2e-run")
        assert len(decisions) == 1
        assert decisions[0]["included"] is True

        checkpoint = store.get_latest_checkpoint("e2e-run")
        assert checkpoint["node"] == "completed"

    def test_list_runs_returns_persisted_runs(self, mock_llm_provider, store):
        """List runs includes persisted runs."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        # Create multiple runs
        for i, status in enumerate([RunStatus.COMPLETED, RunStatus.INSUFFICIENT_EVIDENCE]):
            mock_state = CDRState(
                run_id=f"list-run-{i}",
                question=f"Question {i}",
                status=status,
            )
            with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
                asyncio.run(runner.run(f"Question {i}", run_id=f"list-run-{i}"))

        # List runs
        runs = store.list_runs()
        assert len(runs) >= 2

        run_ids = [r["run_id"] for r in runs]
        assert "list-run-0" in run_ids
        assert "list-run-1" in run_ids

    def test_resume_from_checkpoint(self, mock_llm_provider, store):
        """Checkpoint can be used to resume run (data retrieval test)."""
        from cdr.orchestration.graph import CDRRunner

        mock_state = CDRState(
            run_id="resume-run",
            question="Resume question",
            status=RunStatus.COMPLETED,
        )

        runner = CDRRunner(llm_provider=mock_llm_provider, run_store=store)

        with patch.object(runner.graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
            asyncio.run(runner.run("Resume question", run_id="resume-run"))

        # Retrieve checkpoint
        checkpoint = store.get_latest_checkpoint("resume-run")
        assert checkpoint is not None

        # Verify state can be reconstructed
        restored_state = CDRState.model_validate(checkpoint["state"])
        assert restored_state.run_id == "resume-run"
        assert restored_state.status == RunStatus.COMPLETED

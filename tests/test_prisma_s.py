"""
Tests for PRISMA-S search reporting compliance.

HIGH-4 fix: Ensure executed searches are tracked for reproducibility.
Refs: PRISMA-S (BMJ 2021), CDR_Integral_Audit_2026-01-20.md HIGH-4

PRISMA-S requires:
1. Database name
2. Exact search strategy executed
3. Date of search
4. Number of results
5. Any modifications made to the original query
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cdr.core.schemas import ExecutedSearch, CDRState, SearchPlan, PICO


class TestExecutedSearchSchema:
    """Test ExecutedSearch schema validation."""

    def test_valid_executed_search(self):
        """Valid ExecutedSearch with all fields."""
        search = ExecutedSearch(
            database="PubMed",
            query_planned="diabetes AND metformin AND RCT",
            query_executed="diabetes AND metformin AND RCT",
            executed_at=datetime.now(timezone.utc),
            results_count=150,
            results_fetched=100,
            notes=None,
        )
        assert search.database == "PubMed"
        assert search.results_count == 150
        assert search.results_fetched == 100

    def test_executed_search_with_query_modification(self):
        """Track when query is modified (e.g., CT.gov truncation)."""
        search = ExecutedSearch(
            database="ClinicalTrials.gov",
            query_planned="diabetes mellitus type 2 AND metformin hydrochloride AND randomized controlled trial",
            query_executed="diabetes mellitus type 2",
            executed_at=datetime.now(timezone.utc),
            results_count=50,
            results_fetched=50,
            notes="Query truncated to 6 words for CT.gov compatibility",
        )
        # PRISMA-S: Must document query modifications
        assert search.query_planned != search.query_executed
        assert "truncated" in search.notes

    def test_executed_search_is_immutable(self):
        """Executed searches should be immutable for audit trail."""
        search = ExecutedSearch(
            database="PubMed",
            query_planned="test",
            query_executed="test",
            results_count=10,
            results_fetched=10,
        )
        with pytest.raises(Exception):  # Pydantic frozen model
            search.database = "modified"

    def test_executed_search_validates_counts(self):
        """Results counts must be non-negative."""
        with pytest.raises(ValueError):
            ExecutedSearch(
                database="PubMed",
                query_executed="test",
                results_count=-5,
                results_fetched=0,
            )

    def test_executed_search_with_error_note(self):
        """Track errors in notes for audit trail."""
        search = ExecutedSearch(
            database="PubMed",
            query_planned="test query",
            query_executed="test query",
            results_count=0,
            results_fetched=0,
            notes="Search failed: Connection timeout",
        )
        assert "failed" in search.notes


class TestCDRStateExecutedSearches:
    """Test CDRState integration with executed_searches."""

    def test_cdr_state_has_executed_searches_field(self):
        """CDRState should have executed_searches field."""
        state = CDRState(run_id="test-123", question="test?")
        assert hasattr(state, "executed_searches")
        assert state.executed_searches == []

    def test_cdr_state_stores_multiple_executed_searches(self):
        """CDRState should store multiple executed searches."""
        pubmed_search = ExecutedSearch(
            database="PubMed",
            query_executed="diabetes",
            results_count=100,
            results_fetched=50,
        )
        ct_search = ExecutedSearch(
            database="ClinicalTrials.gov",
            query_executed="diabetes",
            results_count=30,
            results_fetched=30,
        )
        state = CDRState(
            run_id="test-456",
            question="test?",
            executed_searches=[pubmed_search, ct_search],
        )
        assert len(state.executed_searches) == 2
        assert state.executed_searches[0].database == "PubMed"
        assert state.executed_searches[1].database == "ClinicalTrials.gov"


class TestRetrieveNodePRISMAS:
    """Test retrieve_node tracks executed searches for PRISMA-S."""

    @pytest.fixture
    def mock_search_plan(self):
        """Create a mock search plan."""
        return SearchPlan(
            pico=PICO(
                population="patients with diabetes",
                intervention="metformin",
                comparison="placebo",
                outcome="glycemic control",
            ),
            pubmed_query="diabetes AND metformin",
            ct_gov_query="diabetes mellitus type 2 metformin hydrochloride randomized controlled trial study",
            max_results_per_source=100,
        )

    @pytest.fixture
    def mock_state(self, mock_search_plan):
        """Create a mock CDRState with search plan."""
        return CDRState(
            run_id="test-retrieve-789",
            question="Is metformin effective for diabetes?",
            search_plan=mock_search_plan,
        )

    def test_retrieve_node_returns_executed_searches(self, mock_state):
        """retrieve_node should return executed_searches in output."""
        from cdr.orchestration.graph import retrieve_node
        from unittest.mock import MagicMock

        # Mock PubMed client
        mock_pubmed_result = MagicMock()
        mock_pubmed_result.total_count = 150
        mock_pubmed_result.pmids = ["12345", "67890"]

        # Mock CT client result
        mock_ct_result = MagicMock()
        mock_ct_result.total_count = 30
        mock_ct_result.nct_ids = ["NCT00001"]

        config = {"configurable": {"max_results": 50}}

        with (
            patch("cdr.retrieval.pubmed_client.PubMedClient") as mock_pubmed_cls,
            patch("cdr.retrieval.ct_client.ClinicalTrialsClient") as mock_ct_cls,
        ):
            # Setup PubMed mock
            mock_pubmed = MagicMock()
            mock_pubmed.search.return_value = mock_pubmed_result
            mock_pubmed.fetch_records.return_value = []
            mock_pubmed_cls.return_value = mock_pubmed

            # Setup CT mock
            mock_ct = MagicMock()
            mock_ct.search.return_value = mock_ct_result
            mock_ct.fetch_studies.return_value = []
            mock_ct_cls.return_value = mock_ct

            result = asyncio.run(retrieve_node(mock_state, config))

            # Check executed_searches is in output
            assert "executed_searches" in result
            executed = result["executed_searches"]
            assert len(executed) == 2

            # Check PubMed search tracked
            pubmed_exec = next(s for s in executed if s.database == "PubMed")
            assert pubmed_exec.query_executed == "diabetes AND metformin"
            assert pubmed_exec.results_count == 150

            # Check CT.gov search tracked — query < 500 chars passes through unchanged
            ct_exec = next(s for s in executed if s.database == "ClinicalTrials.gov")
            assert ct_exec.query_executed == mock_state.search_plan.ct_gov_query
            assert ct_exec.query_planned == mock_state.search_plan.ct_gov_query
            # No truncation note for queries < 500 chars
            assert ct_exec.notes is None or "truncat" not in (ct_exec.notes or "").lower()

    def test_retrieve_node_empty_search_plan(self):
        """retrieve_node with no search plan returns empty executed_searches."""
        from cdr.orchestration.graph import retrieve_node

        state = CDRState(run_id="test-empty-plan", question="test?", search_plan=None)
        config = {"configurable": {}}

        result = asyncio.run(retrieve_node(state, config))

        assert "executed_searches" in result
        assert result["executed_searches"] == []

    def test_retrieve_node_tracks_query_truncation(self, mock_state):
        """retrieve_node should only flag truncation for queries > 500 chars.

        Queries < 500 chars are passed through to ct_client._sanitize_query()
        without truncation at the node level. Only queries exceeding the
        500-char API limit get a truncation note.
        """
        from cdr.orchestration.graph import retrieve_node

        # The mock CT.gov query is >50 chars but <500 chars — should NOT be truncated
        ct_query = mock_state.search_plan.ct_gov_query
        assert len(ct_query) > 50
        assert len(ct_query) < 500

        config = {"configurable": {"max_results": 10}}

        with (
            patch("cdr.retrieval.pubmed_client.PubMedClient") as mock_pubmed_cls,
            patch("cdr.retrieval.ct_client.ClinicalTrialsClient") as mock_ct_cls,
        ):
            mock_pubmed = MagicMock()
            mock_pubmed.search.return_value = MagicMock(total_count=0, pmids=[])
            mock_pubmed_cls.return_value = mock_pubmed

            mock_ct = MagicMock()
            mock_ct.search.return_value = MagicMock(total_count=0, nct_ids=[])
            mock_ct_cls.return_value = mock_ct

            result = asyncio.run(retrieve_node(mock_state, config))

            ct_exec = next(
                s for s in result["executed_searches"] if s.database == "ClinicalTrials.gov"
            )

            # Query < 500 chars: passed through unchanged, no truncation note
            assert ct_exec.query_executed == ct_query
            assert ct_exec.query_planned == ct_query
            # No truncation note
            assert ct_exec.notes is None or "truncat" not in (ct_exec.notes or "").lower()


class TestReportDataPRISMAS:
    """Test report_data includes executed searches for PRISMA-S audit."""

    def test_report_data_structure(self):
        """Verify report_data includes all PRISMA-S required fields."""
        # This tests the expected structure; actual integration is in test_integration_publish.py
        required_fields = [
            "search_plan",  # Planned search strategy
            "executed_searches",  # Actual executed searches
        ]

        # Mock report_data as produced by publish_node
        report_data = {
            "run_id": "test-123",
            "search_plan": {
                "pubmed_query": "diabetes AND metformin",
                "ct_gov_query": "diabetes metformin",
                "date_range": None,
                "languages": ["en"],
                "max_results_per_source": 100,
                "created_at": "2024-01-01T00:00:00Z",
            },
            "executed_searches": [
                {
                    "database": "PubMed",
                    "query_planned": "diabetes AND metformin",
                    "query_executed": "diabetes AND metformin",
                    "executed_at": "2024-01-01T00:01:00Z",
                    "results_count": 150,
                    "results_fetched": 100,
                    "notes": None,
                },
                {
                    "database": "ClinicalTrials.gov",
                    "query_planned": "diabetes metformin randomized",
                    "query_executed": "diabetes metformin",
                    "executed_at": "2024-01-01T00:01:05Z",
                    "results_count": 30,
                    "results_fetched": 30,
                    "notes": "Query truncated to 6 words for CT.gov compatibility",
                },
            ],
        }

        for field in required_fields:
            assert field in report_data, f"Missing PRISMA-S field: {field}"

        # Verify executed_searches structure
        for search in report_data["executed_searches"]:
            assert "database" in search
            assert "query_executed" in search
            assert "executed_at" in search
            assert "results_count" in search

    def test_executed_search_preserves_original_query(self):
        """Verify we can compare planned vs executed queries."""
        executed = {
            "query_planned": "very long query with many terms that exceeds 50 chars limit",
            "query_executed": "very long query with",
            "notes": "Query truncated",
        }

        # PRISMA-S: Auditor must be able to see what was planned vs executed
        assert executed["query_planned"] != executed["query_executed"]
        assert len(executed["query_executed"]) < len(executed["query_planned"])


class TestPRISMAExclusionReasons:
    """Test PRISMA 2020 exclusion reasons tracking (MEDIUM-2).

    Per PRISMA 2020 Statement:
    "For each record excluded, a reason for exclusion should be recorded."

    Refs: PRISMA 2020, CDR_Integral_Audit_2026-01-20.md MEDIUM-2
    """

    def test_prisma_counts_has_exclusion_reasons_field(self):
        """PRISMACounts should have exclusion_reasons field."""
        from cdr.core.schemas import PRISMACounts

        prisma = PRISMACounts()
        assert hasattr(prisma, "exclusion_reasons")
        assert prisma.exclusion_reasons == {}

    def test_prisma_counts_with_exclusion_breakdown(self):
        """PRISMACounts should store exclusion reasons breakdown."""
        from cdr.core.schemas import PRISMACounts

        exclusion_breakdown = {
            "PICO_MISMATCH": 15,
            "NO_ABSTRACT": 8,
            "ANIMAL_STUDY": 3,
            "STUDY_TYPE_EXCLUDED": 2,
        }

        prisma = PRISMACounts(
            records_screened=100,
            records_excluded=28,
            studies_included=72,
            exclusion_reasons=exclusion_breakdown,
        )

        assert prisma.exclusion_reasons["PICO_MISMATCH"] == 15
        assert prisma.exclusion_reasons["NO_ABSTRACT"] == 8
        assert sum(prisma.exclusion_reasons.values()) == 28
        assert sum(prisma.exclusion_reasons.values()) == prisma.records_excluded

    def test_exclusion_reasons_in_model_dump(self):
        """Exclusion reasons should be included in model_dump for serialization."""
        from cdr.core.schemas import PRISMACounts

        prisma = PRISMACounts(
            records_excluded=10,
            exclusion_reasons={"PICO_MISMATCH": 7, "OTHER": 3},
        )

        dump = prisma.model_dump()

        assert "exclusion_reasons" in dump
        assert dump["exclusion_reasons"]["PICO_MISMATCH"] == 7
        assert dump["exclusion_reasons"]["OTHER"] == 3

    def test_screening_decision_has_reason_code(self):
        """ScreeningDecision should have reason_code for exclusions."""
        from cdr.core.schemas import ScreeningDecision
        from cdr.core.enums import ExclusionReason

        # Exclusion decision with reason
        decision = ScreeningDecision(
            record_id="test-123",
            included=False,
            reason_code=ExclusionReason.PICO_MISMATCH,
            reason_text="Population does not match PICO",
        )

        assert decision.reason_code == ExclusionReason.PICO_MISMATCH
        assert "Population" in decision.reason_text

    def test_exclusion_reason_enum_values(self):
        """Verify ExclusionReason enum has expected values."""
        from cdr.core.enums import ExclusionReason

        # Key exclusion reasons per PRISMA 2020 (lowercase values)
        expected_reasons = [
            "pico_mismatch",
            "no_abstract",
            "animal_study",
            "study_type_excluded",
            "duplicate",
            "other",
        ]

        actual_values = [e.value for e in ExclusionReason]

        for reason in expected_reasons:
            assert reason in actual_values, f"Missing exclusion reason: {reason}"

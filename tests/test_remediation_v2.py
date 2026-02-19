"""
Tests for CDR remediation v2 changes.

Validates:
1. MetricsHelper.get_all() correctly sums Counter values (was using _value)
2. CT.gov query delegation to client (500-char threshold vs old 50-char)
3. run_partial() sequential node execution
4. Fulltext retrieval enabled by default in parse_documents_node
5. RoB2 Methods section prioritization in assess_rob2_node
6. Node imports from cdr.orchestration.nodes package
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from cdr.core.enums import GraphNode, RunStatus
from cdr.core.schemas import CDRState, PICO, SearchPlan


def _make_pico(**overrides) -> PICO:
    """Helper to create a valid PICO for tests."""
    defaults = {
        "population": "Adults with type 2 diabetes",
        "intervention": "Metformin",
        "comparator": "Placebo",
        "outcome": "HbA1c reduction",
    }
    defaults.update(overrides)
    return PICO(**defaults)


def _make_search_plan(**overrides) -> SearchPlan:
    """Helper to create a valid SearchPlan for tests."""
    defaults = {
        "pico": _make_pico(),
        "pubmed_query": "diabetes AND metformin",
        "ct_gov_query": "diabetes mellitus type 2 metformin",
    }
    defaults.update(overrides)
    return SearchPlan(**defaults)


# =============================================================================
# 1. MetricsHelper Counter bug fix
# =============================================================================


class TestMetricsHelperCounterFix:
    """Validate that MetricsHelper.get_all() correctly sums multi-label counters."""

    def test_counter_get_all_no_labels(self):
        """Counter with no label increments should return 0 total."""
        from cdr.observability.metrics import Counter, MetricsHelper

        c = Counter("test_no_labels", "test counter")
        helper = MetricsHelper()
        helper._counters["test_no_labels"] = c

        result = helper.get_all()
        assert result["test_no_labels"] == 0.0

    def test_counter_get_all_single_label(self):
        """Counter incremented with one label combination."""
        from cdr.observability.metrics import Counter, MetricsHelper

        c = Counter("test_single_label", "test counter")
        c.inc(labels={"node": "parse_question"})
        c.inc(labels={"node": "parse_question"})

        helper = MetricsHelper()
        helper._counters["test_single_label"] = c

        result = helper.get_all()
        assert result["test_single_label"] == 2.0

    def test_counter_get_all_multiple_labels(self):
        """Counter incremented across multiple label combinations sums all."""
        from cdr.observability.metrics import Counter, MetricsHelper

        c = Counter("test_multi_label", "test counter")
        c.inc(labels={"node": "retrieve"})  # +1
        c.inc(labels={"node": "retrieve"})  # +1
        c.inc(labels={"node": "screen"})  # +1
        c.inc(labels={"node": "publish"})  # +1

        helper = MetricsHelper()
        helper._counters["test_multi_label"] = c

        result = helper.get_all()
        # Should be 4.0 (sum across all label combos), not AttributeError
        assert result["test_multi_label"] == 4.0

    def test_counter_get_all_inc_by_custom_value(self):
        """Counter incremented by custom amounts sums correctly."""
        from cdr.observability.metrics import Counter, MetricsHelper

        c = Counter("test_custom_val", "test counter")
        c.inc(value=3.0, labels={"source": "pubmed"})
        c.inc(value=7.0, labels={"source": "ct_gov"})

        helper = MetricsHelper()
        helper._counters["test_custom_val"] = c

        result = helper.get_all()
        assert result["test_custom_val"] == 10.0


# =============================================================================
# 2. CT.gov query delegation (500-char threshold)
# =============================================================================


class TestCTGovQueryDelegation:
    """Validate that CT.gov queries < 500 chars are NOT truncated at node level."""

    def test_short_query_passes_through(self):
        """Queries < 500 chars should be passed unchanged to the CT client."""
        from cdr.orchestration.graph import retrieve_node

        short_query = "diabetes mellitus type 2 metformin"
        assert len(short_query) < 500

        state = CDRState(
            run_id="test-short-query",
            question="test?",
            search_plan=_make_search_plan(ct_gov_query=short_query),
        )
        config = {"configurable": {"max_results": 5}}

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

            result = asyncio.run(retrieve_node(state, config))

            ct_exec = next(
                s for s in result["executed_searches"] if s.database == "ClinicalTrials.gov"
            )
            assert ct_exec.query_executed == short_query
            assert ct_exec.notes is None or "truncat" not in (ct_exec.notes or "").lower()

    def test_long_query_flags_truncation(self):
        """Queries > 500 chars should be flagged as truncated."""
        from cdr.orchestration.graph import retrieve_node

        long_query = "diabetes " * 100  # ~900 chars
        assert len(long_query.strip()) > 500

        state = CDRState(
            run_id="test-long-query",
            question="test?",
            search_plan=_make_search_plan(ct_gov_query=long_query.strip()),
        )
        config = {"configurable": {"max_results": 5}}

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

            result = asyncio.run(retrieve_node(state, config))

            ct_exec = next(
                s for s in result["executed_searches"] if s.database == "ClinicalTrials.gov"
            )
            # Should have truncation note
            assert ct_exec.notes is not None
            assert "truncat" in ct_exec.notes.lower()


# =============================================================================
# 3. run_partial() implementation
# =============================================================================


class TestRunPartial:
    """Validate run_partial() sequential node execution."""

    def test_run_partial_invalid_start_node(self):
        """run_partial raises ValueError for unknown start_node."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=MagicMock(), model="test")
        state = CDRState(run_id="test", question="test?")

        with pytest.raises((ValueError, TypeError)):
            asyncio.run(runner.run_partial(state, "nonexistent_node"))

    def test_run_partial_invalid_end_before_start(self):
        """run_partial raises ValueError when end_node precedes start_node."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=MagicMock(), model="test")
        state = CDRState(run_id="test", question="test?")

        with pytest.raises(ValueError, match="precedes start_node"):
            asyncio.run(runner.run_partial(state, GraphNode.SYNTHESIZE, GraphNode.RETRIEVE))

    def test_run_partial_single_node(self):
        """run_partial can execute a single node."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=MagicMock(), model="test")
        state = CDRState(run_id="test-partial", question="test?")

        # Mock node to return empty updates (no PICO validation issues)
        with patch(
            "cdr.orchestration.graph.parse_question_node",
            return_value={},
        ) as mock_node:
            result = asyncio.run(
                runner.run_partial(state, GraphNode.PARSE_QUESTION, GraphNode.PARSE_QUESTION)
            )

            mock_node.assert_called_once()
            assert result.run_id == "test-partial"

    def test_run_partial_with_run_store(self):
        """run_partial saves checkpoints when run_store is configured."""
        from cdr.orchestration.graph import CDRRunner

        mock_store = MagicMock()
        runner = CDRRunner(llm_provider=MagicMock(), model="test", run_store=mock_store)
        state = CDRState(run_id="test-checkpoint", question="test?")

        with patch(
            "cdr.orchestration.graph.parse_question_node",
            return_value={},
        ):
            asyncio.run(
                runner.run_partial(state, GraphNode.PARSE_QUESTION, GraphNode.PARSE_QUESTION)
            )

            # Should have saved a checkpoint
            mock_store.save_checkpoint.assert_called_once()
            call_args = mock_store.save_checkpoint.call_args
            assert call_args[0][0] == "test-checkpoint"  # run_id
            assert call_args[0][1] == "parse_question"  # node name

    def test_run_partial_multiple_nodes(self):
        """run_partial executes nodes in sequence and accumulates state."""
        from cdr.orchestration.graph import CDRRunner

        runner = CDRRunner(llm_provider=MagicMock(), model="test")
        state = CDRState(run_id="test-multi", question="test?")

        call_order = []

        async def mock_parse_question(state, config):
            call_order.append("parse_question")
            return {}

        async def mock_plan_search(state, config):
            call_order.append("plan_search")
            return {}

        with (
            patch("cdr.orchestration.graph.parse_question_node", mock_parse_question),
            patch("cdr.orchestration.graph.plan_search_node", mock_plan_search),
        ):
            asyncio.run(runner.run_partial(state, GraphNode.PARSE_QUESTION, GraphNode.PLAN_SEARCH))

            assert call_order == ["parse_question", "plan_search"]


# =============================================================================
# 4. Node package imports
# =============================================================================


class TestNodePackageImports:
    """Validate that all 13 node functions are importable from the nodes package."""

    def test_all_nodes_importable_from_package(self):
        """All 13 node functions should be importable from cdr.orchestration.nodes."""
        from cdr.orchestration import nodes

        expected_names = [
            "parse_question_node",
            "plan_search_node",
            "retrieve_node",
            "deduplicate_node",
            "screen_node",
            "parse_documents_node",
            "extract_data_node",
            "assess_rob2_node",
            "synthesize_node",
            "critique_node",
            "verify_node",
            "compose_node",
            "publish_node",
        ]

        for name in expected_names:
            fn = getattr(nodes, name, None)
            assert fn is not None, f"{name} should be in cdr.orchestration.nodes"
            assert callable(fn), f"{name} should be callable"

    def test_nodes_importable_from_graph(self):
        """Node functions should still be importable from graph.py for backward compat."""
        from cdr.orchestration import graph

        for name in ["parse_question_node", "publish_node", "retrieve_node"]:
            fn = getattr(graph, name, None)
            assert fn is not None, f"{name} should be in cdr.orchestration.graph"
            assert callable(fn)

    def test_conditional_edges_still_in_graph(self):
        """Conditional edge functions remain in graph.py."""
        from cdr.orchestration.graph import (
            should_continue_after_retrieve,
            should_continue_after_screen,
            should_verify,
        )

        assert callable(should_continue_after_retrieve)
        assert callable(should_continue_after_screen)
        assert callable(should_verify)

    def test_cdr_runner_still_in_graph(self):
        """CDRRunner class remains in graph.py."""
        from cdr.orchestration.graph import CDRRunner

        assert CDRRunner is not None
        runner = CDRRunner(llm_provider=MagicMock(), model="test")
        assert runner.dod_level == 1

    def test_build_cdr_graph_still_in_graph(self):
        """build_cdr_graph() function remains in graph.py."""
        from cdr.orchestration.graph import build_cdr_graph

        assert callable(build_cdr_graph)


# =============================================================================
# 5. Fulltext retrieval default enabled
# =============================================================================


class TestFulltextRetrievalDefault:
    """Validate that fulltext retrieval is enabled by default in parse_documents_node."""

    def test_fulltext_enabled_by_default(self):
        """parse_documents_node should attempt fulltext retrieval by default."""
        import inspect
        from cdr.orchestration.nodes.screening_nodes import parse_documents_node

        source = inspect.getsource(parse_documents_node)
        # Should default to True, not False
        assert (
            'enable_fulltext_retrieval", True)' in source
            or 'enable_fulltext_retrieval", True)' in source
        )


# =============================================================================
# 6. RoB2 Methods section prioritization
# =============================================================================


class TestRoB2MethodsPrioritization:
    """Validate that assess_rob2_node prioritizes Methods section when available."""

    def test_rob2_node_uses_sections_when_available(self):
        """assess_rob2_node should use structured sections for RoB2 assessment."""
        import inspect
        from cdr.orchestration.nodes.analysis_nodes import assess_rob2_node

        source = inspect.getsource(assess_rob2_node)
        # Should reference Methods section
        assert "methods" in source.lower()
        # Should have logic for structured sections
        assert "sections" in source.lower()


# =============================================================================
# 7. Conditional edge correctness after refactor
# =============================================================================


class TestConditionalEdgesPostRefactor:
    """Validate conditional edge functions work correctly after graph split."""

    def test_should_continue_after_retrieve_no_records(self):
        """Route to publish when no records found."""
        from cdr.orchestration.graph import should_continue_after_retrieve

        state = CDRState(
            run_id="test",
            question="test?",
            retrieved_records=[],
        )
        assert should_continue_after_retrieve(state) == "publish"

    def test_should_continue_after_retrieve_with_records(self):
        """Route to deduplicate when records found."""
        from cdr.orchestration.graph import should_continue_after_retrieve
        from cdr.core.schemas import Record

        record = Record(
            record_id="r1",
            source="pubmed",
            content_hash="abc123",
            title="Test",
            abstract="Test abstract",
        )
        state = CDRState(
            run_id="test",
            question="test?",
            retrieved_records=[record],
        )
        assert should_continue_after_retrieve(state) == "deduplicate"

    def test_should_continue_after_screen_no_included(self):
        """Route to publish when no records pass screening."""
        from cdr.orchestration.graph import should_continue_after_screen

        state = CDRState(
            run_id="test",
            question="test?",
            screened=[],  # Empty or all excluded
        )
        # CDRState.get_included_records() returns empty for no screened
        assert should_continue_after_screen(state) == "publish"

    def test_should_verify_no_claims(self):
        """Route to publish when no claims to verify."""
        from cdr.orchestration.graph import should_verify

        state = CDRState(
            run_id="test",
            question="test?",
            claims=[],
        )
        assert should_verify(state) == "publish"

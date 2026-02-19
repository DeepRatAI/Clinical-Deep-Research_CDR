"""
CDR Orchestration Layer

LangGraph-based workflow orchestration for the systematic review pipeline.

Node functions are organized in the `nodes/` package:
- retrieval_nodes: parse_question, plan_search, retrieve, deduplicate
- screening_nodes: screen, parse_documents
- analysis_nodes: extract_data, assess_rob2
- synthesis_nodes: synthesize, critique, verify, compose
- publish_node: publish
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cdr.core.enums import GraphNode, RunStatus
from cdr.core.schemas import CDRState
from cdr.observability.tracer import tracer
from cdr.observability.metrics import metrics

# Import all node functions from the nodes package
from cdr.orchestration.nodes import (
    parse_question_node,
    plan_search_node,
    retrieve_node,
    deduplicate_node,
    screen_node,
    parse_documents_node,
    extract_data_node,
    assess_rob2_node,
    synthesize_node,
    critique_node,
    verify_node,
    compose_node,
    publish_node,
)

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider
    from cdr.storage.run_store import RunStore


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================


def should_continue_after_retrieve(state: CDRState) -> Literal["deduplicate", "publish"]:
    """Check if retrieval produced results.

    CRITICAL: On no records, route to publish (not END) for proper negative outcome
    reporting with PRISMA counts and INSUFFICIENT_EVIDENCE status.
    Refs: PRISMA 2020, CDR_Integral_Audit_2026-01-20.md CRITICAL-1
    """
    if not state.retrieved_records:
        return "publish"  # Force publish for negative outcome reporting
    return "deduplicate"


def should_continue_after_screen(state: CDRState) -> Literal["parse_docs", "publish"]:
    """Check if screening produced included records.

    CRITICAL: On no included records, route to publish (not END) for proper
    negative outcome reporting with exclusion reasons and INSUFFICIENT_EVIDENCE.
    Refs: PRISMA 2020, CDR_Integral_Audit_2026-01-20.md CRITICAL-1
    """
    if not state.get_included_records():
        return "publish"  # Force publish for negative outcome reporting
    return "parse_docs"


def should_verify(state: CDRState) -> Literal["verify", "publish"]:
    """Decide whether to run verification."""
    # Skip verification if no claims
    if not state.claims:
        return "publish"

    # Check if critique has blockers (critical issues)
    if state.critique and state.critique.has_blockers():
        # Still verify to document issues
        return "verify"

    return "verify"


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_cdr_graph() -> CompiledStateGraph:
    """Build the CDR workflow graph.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph with CDRState
    graph = StateGraph(CDRState)

    # Add nodes
    graph.add_node(GraphNode.PARSE_QUESTION.value, parse_question_node)
    graph.add_node(GraphNode.PLAN_SEARCH.value, plan_search_node)
    graph.add_node(GraphNode.RETRIEVE.value, retrieve_node)
    graph.add_node(GraphNode.DEDUPLICATE.value, deduplicate_node)
    graph.add_node(GraphNode.SCREEN.value, screen_node)
    graph.add_node(GraphNode.PARSE_DOCS.value, parse_documents_node)
    graph.add_node(GraphNode.EXTRACT_DATA.value, extract_data_node)
    graph.add_node(GraphNode.ASSESS_ROB2.value, assess_rob2_node)
    graph.add_node(GraphNode.SYNTHESIZE.value, synthesize_node)
    graph.add_node(GraphNode.CRITIQUE.value, critique_node)
    graph.add_node(GraphNode.VERIFY.value, verify_node)
    graph.add_node(GraphNode.COMPOSE.value, compose_node)  # HIGH-1: Compositional inference
    graph.add_node(GraphNode.PUBLISH.value, publish_node)

    # Add edges
    graph.add_edge(START, GraphNode.PARSE_QUESTION.value)
    graph.add_edge(GraphNode.PARSE_QUESTION.value, GraphNode.PLAN_SEARCH.value)
    graph.add_edge(GraphNode.PLAN_SEARCH.value, GraphNode.RETRIEVE.value)

    # Conditional: check if retrieval succeeded
    # CRITICAL: On failure, route to PUBLISH (not END) for negative outcome reporting
    # Refs: PRISMA 2020, CDR_Integral_Audit_2026-01-20.md CRITICAL-1
    graph.add_conditional_edges(
        GraphNode.RETRIEVE.value,
        should_continue_after_retrieve,
        {
            "deduplicate": GraphNode.DEDUPLICATE.value,
            "publish": GraphNode.PUBLISH.value,  # Force publish for INSUFFICIENT_EVIDENCE
        },
    )

    graph.add_edge(GraphNode.DEDUPLICATE.value, GraphNode.SCREEN.value)

    # Conditional: check if screening produced results
    # CRITICAL: On failure, route to PUBLISH (not END) for negative outcome reporting
    # Refs: PRISMA 2020, CDR_Integral_Audit_2026-01-20.md CRITICAL-1
    graph.add_conditional_edges(
        GraphNode.SCREEN.value,
        should_continue_after_screen,
        {
            "parse_docs": GraphNode.PARSE_DOCS.value,
            "publish": GraphNode.PUBLISH.value,  # Force publish for INSUFFICIENT_EVIDENCE
        },
    )

    graph.add_edge(GraphNode.PARSE_DOCS.value, GraphNode.EXTRACT_DATA.value)
    graph.add_edge(GraphNode.EXTRACT_DATA.value, GraphNode.ASSESS_ROB2.value)
    graph.add_edge(GraphNode.ASSESS_ROB2.value, GraphNode.SYNTHESIZE.value)
    graph.add_edge(GraphNode.SYNTHESIZE.value, GraphNode.CRITIQUE.value)

    # Conditional: decide on verification
    graph.add_conditional_edges(
        GraphNode.CRITIQUE.value,
        should_verify,
        {
            "verify": GraphNode.VERIFY.value,
            "publish": GraphNode.PUBLISH.value,
        },
    )

    graph.add_edge(GraphNode.VERIFY.value, GraphNode.COMPOSE.value)
    graph.add_edge(GraphNode.COMPOSE.value, GraphNode.PUBLISH.value)  # HIGH-1
    graph.add_edge(GraphNode.PUBLISH.value, END)

    return graph.compile()


# =============================================================================
# RUNNER
# =============================================================================


class CDRRunner:
    """Run the CDR workflow.

    DoD Levels:
    - Level 1 (exploratory): Allows heuristic fallbacks, minimal validation
    - Level 2 (research-grade): Requires LLM screening, structured outputs, verification coverage
    - Level 3 (SOTA-grade): Full compositional inference, quantitative predictions, threat analysis

    Persistence:
    - If run_store is provided, persists run state to SQLite (MEDIUM-6 fix)
    - Records, screening decisions, and checkpoints are stored for audit

    Refs: ADR-005, CDR_Post_ADR003_v3_PostChange_Audit_and_Actions.md,
          CDR_Integral_Audit_2026-01-20.md MEDIUM-6 (persistence layer)
    """

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        model: str = "gpt-4o",
        output_dir: str = "reports",
        dod_level: int = 1,
        run_store: "RunStore | None" = None,
    ) -> None:
        """Initialize runner.

        Args:
            llm_provider: LLM provider for all nodes
            model: Model to use
            output_dir: Directory for output files
            dod_level: Definition of Done level (1=exploratory, 2=research, 3=SOTA)
            run_store: Optional RunStore for persistent storage (MEDIUM-6)
        """
        self.llm_provider = llm_provider
        self.model = model
        self.output_dir = output_dir
        self.dod_level = dod_level
        self.run_store = run_store
        self.graph = build_cdr_graph()

    async def run(
        self,
        research_question: str,
        max_results: int = 100,
        formats: list[str] | None = None,
        dod_level: int | None = None,
        run_id: str | None = None,
    ) -> CDRState:
        """Run the full CDR pipeline.

        Args:
            research_question: The research question to investigate
            max_results: Maximum records to retrieve per source
            formats: Output formats (default: ["markdown", "json"])
            dod_level: Override DoD level for this run (default: use instance level)
            run_id: Optional run ID from API (if None, generates new UUID)

        Returns:
            Final CDRState with all results
        """
        # Use run-specific or instance-level dod_level
        effective_dod_level = dod_level if dod_level is not None else self.dod_level

        with tracer.start_span("cdr.run") as span:
            # ALTO-C fix: Use provided run_id or generate new one
            # Refs: CDR_Integral_Audit_2026-01-20.md ALTO-C (run_id alignment)
            run_id = run_id or str(uuid.uuid4())[:8]
            span.set_attribute("run_id", run_id)
            span.set_attribute("dod_level", effective_dod_level)

            # Initialize state
            initial_state = CDRState(
                run_id=run_id,
                question=research_question,
                status=RunStatus.RUNNING,
            )

            # MEDIUM-6 fix: Persist run creation if run_store is available
            # Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-6 (persistence layer)
            if self.run_store:
                try:
                    pico_dict = {"question": research_question}  # PICO parsed later
                    self.run_store.create_run(
                        run_id=run_id,
                        pico=pico_dict,
                        metadata={
                            "dod_level": effective_dod_level,
                            "max_results": max_results,
                            "model": self.model,
                        },
                    )
                    self.run_store.update_run_status(
                        run_id, RunStatus.RUNNING, current_node="parse_question"
                    )
                except Exception as persist_err:
                    # Log but don't fail the run if persistence fails
                    print(f"[CDRRunner] Warning: persistence failed: {persist_err}")

            # Build config with dod_level for end-to-end enforcement
            # Refs: ADR-005 (DoD end-to-end)
            # Include run_store in config for node-level persistence
            config = {
                "llm_provider": self.llm_provider,
                "model": self.model,
                "max_results": max_results,
                "output_dir": self.output_dir,
                "formats": formats or ["markdown", "json"],
                "dod_level": effective_dod_level,
                "run_store": self.run_store,  # MEDIUM-6: Pass for node-level persistence
            }

            print(f"[CDRRunner] Starting run {run_id} with DoD Level {effective_dod_level}")
            if effective_dod_level >= 2:
                print(
                    f"[CDRRunner] Research-grade: LLM screening required, structured outputs enforced"
                )
            if effective_dod_level >= 3:
                print(
                    f"[CDRRunner] SOTA-grade: Full compositional inference, verification gates active"
                )

            # Run graph
            try:
                result = await self.graph.ainvoke(
                    initial_state,
                    config={"configurable": config},
                )

                # LangGraph returns a dict, convert to CDRState
                if isinstance(result, dict):
                    final_state = CDRState.model_validate(result)
                else:
                    final_state = result

                # CRITICAL: Preserve the scientific status from publish_node
                # Do NOT overwrite to COMPLETED - the pipeline determines scientific outcome
                # Valid final statuses: COMPLETED (publishable), INSUFFICIENT_EVIDENCE, UNPUBLISHABLE
                if final_state.status == RunStatus.RUNNING:
                    # Only upgrade from RUNNING if pipeline didn't set a final status
                    final_state.status = RunStatus.COMPLETED
                    span.set_attribute("status", "completed")
                else:
                    # Preserve the scientific status determined by publish_node
                    span.set_attribute("status", final_state.status.value)

                # MEDIUM-6 fix: Persist final state to run_store
                # Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-6 (persistence layer)
                if self.run_store:
                    try:
                        # Update run status
                        self.run_store.update_run_status(
                            run_id, final_state.status, current_node="completed"
                        )
                        # Persist records if available (field name: retrieved_records)
                        if final_state.retrieved_records:
                            for record in final_state.retrieved_records:
                                self.run_store.add_record(run_id, record.model_dump())
                        # Persist screening decisions if available (field name: screened)
                        if final_state.screened:
                            for sd in final_state.screened:
                                self.run_store.add_screening_decision(
                                    run_id=run_id,
                                    record_id=sd.record_id,
                                    included=sd.included,
                                    exclusion_reason=sd.reason_code.value
                                    if sd.reason_code
                                    else None,
                                    exclusion_rationale=sd.reason_text,
                                    confidence=sd.pico_match_score,
                                )
                        # Save final checkpoint
                        self.run_store.save_checkpoint(
                            run_id, "completed", final_state.model_dump(mode="json")
                        )
                    except Exception as persist_err:
                        print(f"[CDRRunner] Warning: final persistence failed: {persist_err}")

            except Exception as e:
                import traceback

                print(f"[CDRRunner] ERROR: Pipeline failed with exception: {e}")
                print(f"[CDRRunner] Traceback:\n{traceback.format_exc()}")
                span.set_attribute("error", str(e))
                final_state = CDRState(
                    run_id=run_id,
                    question=research_question,
                    status=RunStatus.FAILED,
                    errors=[str(e)],
                )
                # MEDIUM-6 fix: Persist failed state
                if self.run_store:
                    try:
                        self.run_store.update_run_status(
                            run_id, RunStatus.FAILED, current_node="failed", error_message=str(e)
                        )
                    except Exception:
                        pass  # Don't mask original error

            return final_state

    async def run_partial(
        self,
        state: CDRState,
        start_node: GraphNode,
        end_node: GraphNode | None = None,
    ) -> CDRState:
        """Run a partial pipeline from a checkpoint.

        Executes nodes sequentially from start_node through end_node (inclusive).
        Uses direct node function calls rather than LangGraph's graph.ainvoke()
        to support arbitrary subgraph execution without graph interruption.

        If a RunStore is configured, checkpoints are saved after each node completes.

        Args:
            state: Current CDR state to resume from (e.g., from a checkpoint)
            start_node: First node to execute
            end_node: Last node to execute (inclusive). If None, runs to PUBLISH.

        Returns:
            Updated CDRState after executing the node range

        Raises:
            ValueError: If start_node or end_node is not in the pipeline order,
                       or if end_node precedes start_node.
        """
        # Canonical pipeline order (matches graph topology)
        pipeline_order = [
            GraphNode.PARSE_QUESTION,
            GraphNode.PLAN_SEARCH,
            GraphNode.RETRIEVE,
            GraphNode.DEDUPLICATE,
            GraphNode.SCREEN,
            GraphNode.PARSE_DOCS,
            GraphNode.EXTRACT_DATA,
            GraphNode.ASSESS_ROB2,
            GraphNode.SYNTHESIZE,
            GraphNode.CRITIQUE,
            GraphNode.VERIFY,
            GraphNode.COMPOSE,
            GraphNode.PUBLISH,
        ]

        # Map node enum to callable
        node_functions = {
            GraphNode.PARSE_QUESTION: parse_question_node,
            GraphNode.PLAN_SEARCH: plan_search_node,
            GraphNode.RETRIEVE: retrieve_node,
            GraphNode.DEDUPLICATE: deduplicate_node,
            GraphNode.SCREEN: screen_node,
            GraphNode.PARSE_DOCS: parse_documents_node,
            GraphNode.EXTRACT_DATA: extract_data_node,
            GraphNode.ASSESS_ROB2: assess_rob2_node,
            GraphNode.SYNTHESIZE: synthesize_node,
            GraphNode.CRITIQUE: critique_node,
            GraphNode.VERIFY: verify_node,
            GraphNode.COMPOSE: compose_node,
            GraphNode.PUBLISH: publish_node,
        }

        if end_node is None:
            end_node = GraphNode.PUBLISH

        # Validate nodes
        if start_node not in pipeline_order:
            raise ValueError(f"Unknown start_node: {start_node}")
        if end_node not in pipeline_order:
            raise ValueError(f"Unknown end_node: {end_node}")

        start_idx = pipeline_order.index(start_node)
        end_idx = pipeline_order.index(end_node)

        if end_idx < start_idx:
            raise ValueError(
                f"end_node ({end_node.value}) precedes start_node ({start_node.value}) "
                f"in the pipeline order"
            )

        # Build config (same as full run)
        config = {
            "configurable": {
                "llm_provider": self.llm_provider,
                "model": self.model,
                "max_results": 100,
                "output_dir": self.output_dir,
                "formats": ["markdown", "json"],
                "dod_level": self.dod_level,
                "run_store": self.run_store,
            }
        }

        nodes_to_run = pipeline_order[start_idx : end_idx + 1]
        print(
            f"[CDRRunner] Partial run: {start_node.value} → {end_node.value} "
            f"({len(nodes_to_run)} nodes)"
        )

        # Execute nodes sequentially
        current_state = state
        for node_enum in nodes_to_run:
            node_fn = node_functions[node_enum]
            print(f"[CDRRunner] Executing node: {node_enum.value}")

            if self.run_store:
                try:
                    self.run_store.update_run_status(
                        state.run_id, RunStatus.RUNNING, current_node=node_enum.value
                    )
                except Exception:
                    pass  # Don't fail run on persistence error

            # Each node returns a dict of state updates
            updates = await node_fn(current_state, config)

            # Apply updates to state
            if isinstance(updates, dict):
                state_dict = current_state.model_dump()
                state_dict.update(updates)
                current_state = CDRState.model_validate(state_dict)
            elif isinstance(updates, CDRState):
                current_state = updates

            # Save checkpoint
            if self.run_store:
                try:
                    self.run_store.save_checkpoint(
                        state.run_id,
                        node_enum.value,
                        current_state.model_dump(mode="json"),
                    )
                except Exception:
                    pass

        print(f"[CDRRunner] Partial run complete")
        return current_state

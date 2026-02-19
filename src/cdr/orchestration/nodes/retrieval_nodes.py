"""
CDR Retrieval Nodes

Node functions for question parsing, search planning, retrieval, and deduplication.
Extracted from graph.py monolith.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from cdr.core.enums import RecordSource
from cdr.core.schemas import CDRState, PRISMACounts
from cdr.observability.tracer import tracer
from cdr.observability.metrics import metrics


async def parse_question_node(state: CDRState, config: RunnableConfig) -> dict:
    """Parse research question into PICO components.

    Node: PARSE_QUESTION
    Input: question
    Output: pico
    """
    with tracer.start_span("node.parse_question") as span:
        span.set_attribute("question_length", len(state.question))

        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        # Import here to avoid circular imports
        from cdr.interface.question_parser import QuestionParser

        parser = QuestionParser(llm, model)
        pico = await parser.parse(state.question)

        span.set_attribute("pico_parsed", pico is not None)

        return {"pico": pico}


async def plan_search_node(state: CDRState, config: RunnableConfig) -> dict:
    """Generate search plan from PICO.

    Node: PLAN_SEARCH
    Input: pico
    Output: search_plan
    """
    with tracer.start_span("node.plan_search") as span:
        if not state.pico:
            span.set_attribute("error", "No PICO available")
            return {"errors": [*state.errors, "Cannot plan search without PICO"]}

        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        from cdr.interface.search_planner import SearchPlanner

        planner = SearchPlanner(llm, model)
        search_plan = await planner.plan(state.pico)

        span.set_attribute("pubmed_query_length", len(search_plan.pubmed_query or ""))

        return {"search_plan": search_plan}


async def retrieve_node(state: CDRState, config: RunnableConfig) -> dict:
    """Execute retrieval from PubMed and ClinicalTrials.gov.

    Node: RETRIEVE
    Input: search_plan
    Output: retrieved_records, prisma_counts (partial), executed_searches

    HIGH-4 fix: Track executed searches for PRISMA-S compliance.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-4
    """
    with tracer.start_span("node.retrieve") as span:
        # Import schemas needed for this node
        from cdr.core.schemas import PRISMACounts, ExecutedSearch
        from datetime import datetime, timezone

        if not state.search_plan:
            # MEDIO-E fix: Initialize empty PRISMA counts for early failures
            # This ensures report_data always has auditable counts
            # Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-E, PRISMA 2020
            return {
                "errors": [*state.errors, "No search plan available"],
                "prisma_counts": PRISMACounts(),
                "executed_searches": [],
            }

        configurable = config.get("configurable", {})
        max_results = configurable.get("max_results", 100)
        from cdr.retrieval.pubmed_client import PubMedClient
        from cdr.retrieval.ct_client import ClinicalTrialsClient

        pubmed = PubMedClient()
        ct_client = ClinicalTrialsClient()

        records = []
        errors = list(state.errors)
        executed_searches: list[ExecutedSearch] = []

        # PubMed search
        print(f"[Retrieve] PubMed query: {state.search_plan.pubmed_query}")
        pubmed_total_count = 0
        pubmed_fetched_count = 0
        pubmed_error_note = None
        if state.search_plan.pubmed_query:
            try:
                # PubMedClient.search is synchronous, returns PubMedSearchResult
                search_result = pubmed.search(
                    state.search_plan.pubmed_query,
                    max_results=max_results,
                )
                pubmed_total_count = search_result.total_count
                print(
                    f"[Retrieve] PubMed found {search_result.total_count} results, fetching {len(search_result.pmids)} PMIDs"
                )

                # Fetch full records for the PMIDs
                if search_result.pmids:
                    try:
                        pubmed_records = pubmed.fetch_records(search_result.pmids)
                        records.extend(pubmed_records)
                        pubmed_fetched_count = len(pubmed_records)
                        print(f"[Retrieve] Fetched {len(pubmed_records)} PubMed records")
                    except Exception as fetch_error:
                        print(f"[Retrieve] PubMed fetch error: {fetch_error}")
                        errors.append(f"PubMed fetch failed: {fetch_error}")
                        pubmed_error_note = f"Fetch failed: {fetch_error}"

                span.set_attribute("pubmed_count", len(search_result.pmids))
            except Exception as e:
                print(f"[Retrieve] PubMed error: {e}")
                span.set_attribute("pubmed_error", str(e))
                errors.append(f"PubMed search failed: {e}")
                pubmed_error_note = f"Search failed: {e}"

        # Track executed PubMed search for PRISMA-S
        if state.search_plan.pubmed_query:
            executed_searches.append(
                ExecutedSearch(
                    database="PubMed",
                    query_planned=state.search_plan.pubmed_query,
                    query_executed=state.search_plan.pubmed_query,  # No modification for PubMed
                    executed_at=datetime.now(timezone.utc),
                    results_count=pubmed_total_count,
                    results_fetched=pubmed_fetched_count,
                    notes=pubmed_error_note,
                )
            )

        # ClinicalTrials.gov search with structured filters
        print(f"[Retrieve] CT.gov query: {state.search_plan.ct_gov_query}")
        ct_query_executed = None
        ct_total_count = 0
        ct_fetched_count = 0
        ct_error_note = None
        ct_query_truncated = False
        if state.search_plan.ct_gov_query:
            try:
                from cdr.retrieval.ct_client import CTSearchFilters

                # Use structured filters for better results
                # Focus on INTERVENTIONAL studies only (most common filter)
                ct_filters = CTSearchFilters(
                    study_type="INTERVENTIONAL",
                )

                # Simplify query - CT.gov works better with shorter queries
                # Delegate sanitization to ct_client._sanitize_query() which
                # handles up to 500 chars with proper term extraction
                ct_query = state.search_plan.ct_gov_query
                ct_query_truncated = False

                # Only flag as truncated if ct_client will actually truncate
                if len(ct_query) > 500:
                    ct_query_truncated = True
                    print(
                        f"[CT.gov] Long query ({len(ct_query)} chars), will be sanitized by client"
                    )

                ct_query_executed = ct_query

                # CTClient.search is synchronous, returns CTSearchResult
                ct_result = ct_client.search(
                    ct_query,
                    max_results=max_results,
                    filters=ct_filters,
                )
                ct_total_count = (
                    ct_result.total_count
                    if hasattr(ct_result, "total_count")
                    else len(ct_result.nct_ids)
                )
                # Get the records from the result
                ct_records = ct_client.fetch_studies(ct_result.nct_ids) if ct_result.nct_ids else []
                records.extend(ct_records)
                ct_fetched_count = len(ct_records)
                print(f"[Retrieve] CT.gov returned {len(ct_records)} records")
                span.set_attribute("ct_count", len(ct_records))
            except Exception as e:
                print(f"[Retrieve] CT.gov error: {e}")
                span.set_attribute("ct_error", str(e))
                errors.append(f"CT.gov search failed: {e}")
                ct_error_note = f"Search failed: {e}"

        # Track executed CT.gov search for PRISMA-S
        if state.search_plan.ct_gov_query:
            notes_parts = []
            if ct_query_truncated:
                notes_parts.append("Query truncated by CT.gov client (exceeded 500-char limit)")
            if ct_error_note:
                notes_parts.append(ct_error_note)
            executed_searches.append(
                ExecutedSearch(
                    database="ClinicalTrials.gov",
                    query_planned=state.search_plan.ct_gov_query,
                    query_executed=ct_query_executed or state.search_plan.ct_gov_query,
                    executed_at=datetime.now(timezone.utc),
                    results_count=ct_total_count,
                    results_fetched=ct_fetched_count,
                    notes="; ".join(notes_parts) if notes_parts else None,
                )
            )

        # =====================================================================
        # MEDIUM-3: Hybrid retrieval scoring
        # Compute BM25 and optional reranking scores for retrieval quality
        # Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-3, RAG best practices
        # =====================================================================
        enable_hybrid = configurable.get("enable_hybrid_retrieval", True)

        if records and enable_hybrid:
            try:
                from cdr.retrieval.bm25 import BM25Retriever

                # Build BM25 index from abstracts
                doc_ids = [r.record_id for r in records]
                texts = [f"{r.title or ''} {r.abstract or ''}" for r in records]

                bm25 = BM25Retriever()
                bm25.index_documents(doc_ids, texts)

                # Score using PICO-based query
                pico_query = ""
                if state.pico:
                    pico_query = f"{state.pico.population} {state.pico.intervention}"
                    if state.pico.comparator:
                        pico_query += f" {state.pico.comparator}"
                    if state.pico.outcome:
                        pico_query += f" {state.pico.outcome}"
                elif state.search_plan:
                    pico_query = state.search_plan.pubmed_query or ""

                if pico_query:
                    bm25_results = bm25.search(pico_query, top_k=len(records))
                    bm25_scores = {r[0]: r[1] for r in bm25_results}

                    # Update records with BM25 scores
                    scored_records = []
                    for record in records:
                        scores = dict(record.retrieval_scores) if record.retrieval_scores else {}
                        scores["bm25"] = bm25_scores.get(record.record_id, 0.0)
                        # Create new record with updated scores
                        scored_record = record.model_copy(update={"retrieval_scores": scores})
                        scored_records.append(scored_record)

                    records = scored_records
                    print(f"[Retrieve] BM25 scoring applied to {len(records)} records")
                    span.set_attribute("bm25_scored", True)

                # Optional: cross-encoder reranking (expensive, disabled by default)
                enable_rerank = configurable.get("enable_reranker", False)
                if enable_rerank and pico_query:
                    try:
                        from cdr.retrieval.reranker import get_reranker

                        reranker = get_reranker()
                        # Score records with cross-encoder
                        rerank_texts = [
                            f"{r.title or ''}: {(r.abstract or '')[:500]}" for r in records
                        ]
                        rerank_scores = reranker.score_batch(pico_query, rerank_texts)

                        # Add rerank scores
                        reranked_records = []
                        for i, record in enumerate(records):
                            scores = (
                                dict(record.retrieval_scores) if record.retrieval_scores else {}
                            )
                            scores["rerank"] = rerank_scores[i]
                            reranked_record = record.model_copy(update={"retrieval_scores": scores})
                            reranked_records.append(reranked_record)

                        # Sort by rerank score (descending)
                        records = sorted(
                            reranked_records,
                            key=lambda r: r.retrieval_scores.get("rerank", 0),
                            reverse=True,
                        )
                        print(f"[Retrieve] Cross-encoder reranking applied")
                        span.set_attribute("reranked", True)
                    except Exception as rerank_err:
                        print(f"[Retrieve] Reranking skipped: {rerank_err}")
                        span.set_attribute("rerank_error", str(rerank_err))

            except Exception as hybrid_err:
                print(f"[Retrieve] Hybrid scoring skipped: {hybrid_err}")
                span.set_attribute("hybrid_error", str(hybrid_err))

        # Update PRISMA counts
        prisma = PRISMACounts(
            records_identified=len(records),
            records_from_pubmed=len([r for r in records if r.source == RecordSource.PUBMED]),
            records_from_clinical_trials=len(
                [r for r in records if r.source == RecordSource.CLINICAL_TRIALS]
            ),
        )

        span.set_attribute("total_records", len(records))
        metrics.counter("cdr.records.retrieved", len(records))
        print(f"[Retrieve] Total records: {len(records)}")

        return {
            "retrieved_records": records,
            "prisma_counts": prisma,
            "errors": errors,
            "executed_searches": executed_searches,
        }


async def deduplicate_node(state: CDRState, config: RunnableConfig) -> dict:
    """Remove duplicate records.

    Node: DEDUPLICATE
    Input: retrieved_records
    Output: retrieved_records (deduplicated), prisma_counts
    """
    _ = config  # unused but required by LangGraph
    with tracer.start_span("node.deduplicate") as span:
        records = state.retrieved_records
        original_count = len(records)

        # Deduplicate by PMID, DOI, or title similarity
        seen_ids = set()
        deduplicated = []

        for record in records:
            # Use PMID or DOI as primary key
            key = record.pmid or record.doi or record.title.lower()[:50]

            if key not in seen_ids:
                seen_ids.add(key)
                deduplicated.append(record)

        removed = original_count - len(deduplicated)
        span.set_attribute("duplicates_removed", removed)
        print(f"[Deduplicate] Removed {removed} duplicates, {len(deduplicated)} records remaining")

        # Create new PRISMA with updated counts (PRISMACounts is frozen/immutable)
        old_prisma = state.prisma_counts
        prisma = PRISMACounts(
            records_identified=old_prisma.records_identified if old_prisma else original_count,
            records_from_pubmed=old_prisma.records_from_pubmed if old_prisma else 0,
            records_from_clinical_trials=old_prisma.records_from_clinical_trials
            if old_prisma
            else 0,
            duplicates_removed=removed,
            records_screened=len(deduplicated),
        )

        return {
            "retrieved_records": deduplicated,
            "prisma_counts": prisma,
        }

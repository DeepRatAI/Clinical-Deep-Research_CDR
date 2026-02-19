"""
CDR Screening Nodes

Node functions for record screening and document parsing.
Extracted from graph.py monolith.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from cdr.core.schemas import CDRState, PRISMACounts
from cdr.observability.tracer import tracer
from cdr.observability.metrics import metrics
from cdr.orchestration.nodes._helpers import _extract_pico_terms, _calculate_pico_match_score


async def screen_node(state: CDRState, config: RunnableConfig) -> dict:
    """Screen records for inclusion/exclusion based on PICO criteria.

    Node: SCREEN
    Input: retrieved_records, pico
    Output: screened (list of ScreeningDecision)

    IMPLEMENTATION:
    Uses LLM-based Screener for PICO matching when LLM is available.
    Falls back to heuristic screening if no LLM configured AND dod_level == 1.

    For Research-grade (dod_level >= 2), LLM screening is REQUIRED.
    Heuristic fallback for Level 2+ returns INSUFFICIENT_EVIDENCE.

    Screening criteria:
    - Population relevance assessment
    - Intervention/Comparator alignment
    - Outcome relevance check
    - Study design filtering

    Per PRISMA 2020: All screening decisions must be traceable with reasons.
    Refs: ADR-004 Audit v3 (level-gating)
    """
    with tracer.start_span("node.screen") as span:
        if not state.pico:
            return {"errors": [*state.errors, "Cannot screen without PICO"]}

        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")
        # DoD Level: 1 = exploratory, 2 = research-grade, 3 = SOTA-grade
        # Default to 1 for backwards compatibility
        dod_level = configurable.get("dod_level", 1)

        from cdr.core.enums import ExclusionReason
        from cdr.core.schemas import ScreeningDecision

        included_count = 0
        excluded_count = 0
        screened = []

        span.set_attribute("dod_level", dod_level)

        # Use LLM-based screening if available
        if llm:
            from cdr.screening.screener import Screener

            # CRITICAL: Use the injected llm provider from the runner for consistency
            # This ensures determinism and proper observability tracing
            # Refs: LangGraph configurable state, LangChain runnables
            screener = Screener(
                provider=llm,  # Use injected provider, not settings
                model=model,
            )

            print(f"[Screen] Using LLM-based screening for {len(state.retrieved_records)} records")

            for record in state.retrieved_records:
                decision = screener.screen_record(state.pico, record)
                screened.append(decision)
                if decision.included:
                    included_count += 1
                else:
                    excluded_count += 1
        else:
            # Fallback: PICO-informed heuristic screening (less accurate than LLM)
            # ⚠️ LEVEL-GATING: This fallback is ONLY acceptable for Level 1 (exploratory)
            # For Research-grade (Level 2+), LLM or manual review is REQUIRED
            # Refs: PRISMA 2020 Flow Diagram, ADR-004 Audit v3 (level-gating)

            # CRITICAL: Block heuristic screening for Research-grade (Level 2+)
            if dod_level >= 2:
                print(f"[Screen] ❌ ERROR: No LLM available for Research-grade (Level {dod_level})")
                print(f"[Screen] Heuristic screening is insufficient for Level 2+ per PRISMA 2020")
                span.set_attribute("screening_blocked", True)
                span.set_attribute("screening_error", f"LLM_REQUIRED_FOR_LEVEL_{dod_level}")

                # Return error with INSUFFICIENT_EVIDENCE flag
                # This allows the pipeline to fail gracefully with proper PRISMA reason
                return {
                    "errors": [
                        *state.errors,
                        f"INSUFFICIENT_EVIDENCE: LLM screening required for DoD Level {dod_level}. "
                        "Configure LLM provider or reduce to Level 1 (exploratory).",
                    ],
                    "flags": {**state.flags, "screening_blocked_no_llm": True},
                }

            # Level 1 (exploratory) - allow heuristic fallback with warning
            print(
                f"[Screen] ⚠️ WARNING: No LLM available - using heuristic screening (Level 1 only)"
            )
            print(
                f"[Screen] For Research-grade reviews, configure LLM provider or use manual screening"
            )
            span.set_attribute("screening_warning", "heuristic_fallback_level1_only")

            # CRITICAL: Per PRISMA 2020, exclusion must be justified with reason_code
            # This fallback is conservative - includes studies for manual review

            # Extract PICO terms for basic keyword matching
            pico_terms = _extract_pico_terms(state.pico)

            for record in state.retrieved_records:
                # GATE 1: Must have abstract for any assessment
                if not record.abstract:
                    decision = ScreeningDecision(
                        record_id=record.record_id,
                        included=False,
                        reason_code=ExclusionReason.NO_ABSTRACT,
                        reason_text="No abstract available - cannot assess PICO relevance",
                        pico_match_score=0.0,
                    )
                    excluded_count += 1
                    screened.append(decision)
                    continue

                # GATE 2: Basic PICO keyword matching (conservative)
                # Include if abstract mentions ANY PICO component
                abstract_lower = record.abstract.lower()
                title_lower = (record.title or "").lower()
                combined_text = f"{title_lower} {abstract_lower}"

                pico_match_score, components_matched = _calculate_pico_match_score(
                    combined_text, pico_terms
                )

                # CRITICAL: Per PRISMA 2020 / auditoría formal:
                # - Require at least 2 PICO components (P+I or P+O) for inclusion
                # - Threshold raised to 0.4 (was 0.2) to reduce false positives
                # - Single-component matches are weak and inflate PRISMA counts
                # Refs: PRISMA 2020 eligibility, Cochrane Handbook Section 4.6
                if pico_match_score >= 0.4 and components_matched >= 2:
                    decision = ScreeningDecision(
                        record_id=record.record_id,
                        included=True,
                        pico_match_score=pico_match_score,
                        # Note: heuristic screening, needs manual verification
                    )
                    included_count += 1
                else:
                    # Build detailed exclusion reason
                    if components_matched < 2:
                        reason = f"Insufficient PICO coverage ({components_matched}/4 components, score: {pico_match_score:.2f})"
                    else:
                        reason = f"PICO score below threshold (score: {pico_match_score:.2f} < 0.4)"

                    decision = ScreeningDecision(
                        record_id=record.record_id,
                        included=False,
                        reason_code=ExclusionReason.PICO_MISMATCH,
                        reason_text=reason,
                        pico_match_score=pico_match_score,
                    )
                    excluded_count += 1
                screened.append(decision)

        print(f"[Screen] Included: {included_count}, Excluded: {excluded_count}")
        span.set_attribute("included_count", included_count)
        span.set_attribute("excluded_count", excluded_count)
        span.set_attribute("screening_method", "llm" if llm else "heuristic")

        # =====================================================================
        # EVIDENCE GATES: Deterministic validation of included records
        # Run after LLM/heuristic screening to catch:
        # - Population exclusion criteria violations
        # - Comparator mismatches
        # - Study type inconsistencies
        # Refs: PRISMA 2020 eligibility, GRADE indirectness domain
        # =====================================================================
        enable_evidence_gates = configurable.get("enable_evidence_gates", True)

        if enable_evidence_gates and state.pico:
            from cdr.verification.evidence_gates import EvidenceValidator, GateResult

            validator = EvidenceValidator(strict=False)
            gate_excluded_count = 0
            gate_warned_count = 0

            # Create lookup for quick record access
            record_map = {r.record_id: r for r in state.retrieved_records}

            # Process each screened decision
            updated_screened = []
            for decision in screened:
                if not decision.included:
                    # Already excluded, keep as-is
                    updated_screened.append(decision)
                    continue

                # Get record and run gates
                record = record_map.get(decision.record_id)
                if not record:
                    updated_screened.append(decision)
                    continue

                gate_result = validator.validate_record(record, state.pico)

                if gate_result.overall_result == GateResult.FAIL:
                    # Convert to exclusion
                    # Determine specific reason from gate violations
                    reason_code = ExclusionReason.PICO_MISMATCH
                    reason_text = gate_result.degraded_reason or "Evidence gate validation failed"

                    for gr in gate_result.gate_results:
                        if gr.failed and gr.violations:
                            v = gr.violations[0]
                            if v.mismatch_type.value == "population_excluded":
                                reason_code = ExclusionReason.POPULATION_EXCLUDED
                            elif v.mismatch_type.value == "population_not_mentioned":
                                reason_code = ExclusionReason.POPULATION_NOT_IN_SCOPE
                            elif v.mismatch_type.value == "comparator_indirect":
                                reason_code = ExclusionReason.COMPARATOR_INDIRECT
                            elif v.mismatch_type.value == "study_type_mismatch":
                                reason_code = ExclusionReason.STUDY_TYPE_MISMATCH
                            break

                    new_decision = ScreeningDecision(
                        record_id=decision.record_id,
                        included=False,
                        reason_code=reason_code,
                        reason_text=f"[EvidenceGate] {reason_text}",
                        pico_match_score=decision.pico_match_score,
                    )
                    updated_screened.append(new_decision)
                    gate_excluded_count += 1
                    included_count -= 1
                    excluded_count += 1
                    print(
                        f"[Screen] Gate excluded {record.pmid or record.record_id}: {reason_text}"
                    )
                elif gate_result.overall_result == GateResult.WARN:
                    # Keep included but flag in state
                    updated_screened.append(decision)
                    gate_warned_count += 1
                else:
                    updated_screened.append(decision)

            screened = updated_screened

            if gate_excluded_count > 0 or gate_warned_count > 0:
                print(
                    f"[Screen] Evidence gates: {gate_excluded_count} excluded, "
                    f"{gate_warned_count} warned"
                )
                span.set_attribute("gate_excluded", gate_excluded_count)
                span.set_attribute("gate_warned", gate_warned_count)

        # MEDIUM-2 fix: Aggregate exclusion reasons for PRISMA 2020 flow diagram
        # Per PRISMA 2020: "reasons for exclusion should be recorded and reported"
        # Refs: PRISMA 2020 Statement, CDR_Integral_Audit_2026-01-20.md MEDIUM-2
        exclusion_reasons_count: dict[str, int] = {}
        for decision in screened:
            if not decision.included and decision.reason_code:
                reason_key = decision.reason_code.value
                exclusion_reasons_count[reason_key] = exclusion_reasons_count.get(reason_key, 0) + 1

        if exclusion_reasons_count:
            print(f"[Screen] Exclusion breakdown: {exclusion_reasons_count}")
            span.set_attribute("exclusion_breakdown", str(exclusion_reasons_count))

        # Update PRISMA - CRITICAL: Preserve all counts from previous state
        old_prisma = state.prisma_counts
        prisma = PRISMACounts(
            # Preserve identification counts (set in retrieve_node)
            records_identified=old_prisma.records_identified
            if old_prisma
            else len(state.retrieved_records),
            records_from_pubmed=old_prisma.records_from_pubmed if old_prisma else 0,
            records_from_clinical_trials=old_prisma.records_from_clinical_trials
            if old_prisma
            else 0,
            records_from_other=old_prisma.records_from_other if old_prisma else 0,
            duplicates_removed=old_prisma.duplicates_removed if old_prisma else 0,
            # Update screening counts
            records_screened=len(state.retrieved_records),
            records_excluded=excluded_count,
            studies_included=included_count,
            # MEDIUM-2: Include exclusion reasons breakdown for PRISMA 2020
            exclusion_reasons=exclusion_reasons_count,
        )

        metrics.counter("cdr.records.included", included_count)
        metrics.counter("cdr.records.excluded", excluded_count)

        return {
            "screened": screened,
            "prisma_counts": prisma,
        }


async def parse_documents_node(state: CDRState, config: RunnableConfig) -> dict:
    """Parse full-text documents for included records.

    Node: PARSE_DOCS
    Input: included_records
    Output: parsed_documents, snippets, prisma_counts (updated)

    CRITICAL: This node generates REAL Snippets with valid SourceRef.
    These snippets are used by synthesizer and must exist for claim traceability.

    Per PRISMA 2020: Records without retrievable reports (no abstract >= 10 chars)
    are counted in reports_not_retrieved and excluded from synthesis.

    HIGH-2 fix: Attempts PMC full-text retrieval before marking as not_retrieved.
    Refs: PRISMA 2020 Flow Diagram, CDR_Integral_Audit_2026-01-20.md HIGH-2
    """
    configurable = config.get("configurable", {})
    enable_fulltext = configurable.get("enable_fulltext_retrieval", True)

    with tracer.start_span("node.parse_documents") as span:
        from cdr.core.schemas import Snippet, SourceRef, PRISMACounts
        from cdr.core.enums import Section

        parsed = {}
        snippets = []
        reports_not_retrieved = 0
        fulltext_retrieved = 0
        records_with_snippet = []

        included_records = state.get_included_records()
        print(f"[ParseDocs] Processing {len(included_records)} included records")
        span.set_attribute("enable_fulltext", enable_fulltext)

        # Initialize fulltext client if enabled
        fulltext_client = None
        if enable_fulltext:
            try:
                from cdr.retrieval.fulltext_client import FullTextClient

                fulltext_client = FullTextClient()
                print("[ParseDocs] Full-text retrieval enabled (PMC fallback)")
            except ImportError:
                print("[ParseDocs] ⚠️ Full-text client not available, using abstracts only")

        for record in included_records:
            abstract = record.abstract or ""
            document_text = abstract
            text_source = "abstract"
            record_sections = None  # Sections from PMC fulltext if available

            # ================================================================
            # FULLTEXT RETRIEVAL STRATEGY (enhanced for RoB2 quality):
            # 1. Always try PMC fulltext when client available + PMID exists
            # 2. If fulltext found, use it (gives Methods section for RoB2)
            # 3. If fulltext not found, fall back to abstract
            # 4. If neither available, mark as not_retrieved
            # Refs: PRISMA 2020, Cochrane RoB2 (requires Methods section)
            # ================================================================
            if fulltext_client and record.pmid:
                try:
                    ft_result = await fulltext_client.get_full_text(
                        record_id=record.record_id,
                        pmid=record.pmid,
                        abstract=abstract,
                    )
                    if ft_result.source == "pmc_fulltext" and ft_result.full_text:
                        document_text = ft_result.full_text
                        text_source = "pmc_fulltext"
                        record_sections = ft_result.sections
                        fulltext_retrieved += 1
                        print(
                            f"[ParseDocs] ✓ Record {record.record_id}: Retrieved from PMC ({ft_result.pmcid})"
                        )
                    elif ft_result.source == "abstract_fallback" and ft_result.full_text:
                        document_text = ft_result.full_text
                        text_source = "abstract_fallback"
                    # else: keep original abstract as document_text
                except Exception as e:
                    print(f"[ParseDocs] ⚠️ PMC retrieval failed for {record.record_id}: {e}")
                    # Continue with abstract fallback

            # CRITICAL: Check minimum content requirement per PRISMA/ADR-004
            if not document_text or len(document_text) < 10:
                reports_not_retrieved += 1
                print(
                    f"[ParseDocs] ⚠️ Record {record.record_id}: No retrievable content "
                    f"(text length: {len(document_text)}) - counted as 'report not retrieved'"
                )
                continue

            # Use document text (abstract or full-text)
            # Store sections when fulltext provides them for RoB2/synthesis
            parsed[record.record_id] = {
                "text": document_text,
                "source": text_source,
                "title": record.title or "",
                "sections": record_sections,  # Methods, Results, etc. if available
            }
            records_with_snippet.append(record.record_id)

            # CRITICAL: Generate REAL Snippets with valid SourceRef
            # Each snippet represents a citable piece of evidence
            if document_text and len(document_text) >= 10:
                # Determine section based on source
                section = Section.FULL_TEXT if text_source == "pmc_fulltext" else Section.ABSTRACT

                # Create source reference for this record
                source_ref = SourceRef(
                    record_id=record.record_id,
                    pmid=record.pmid,
                    doi=record.doi,
                    section=section,
                    page=None,
                    offset_start=0,
                    offset_end=len(document_text),
                )

                # Create snippet from document text (abstract or full-text)
                snippet = Snippet(
                    snippet_id=f"{record.record_id}_snip_0",
                    text=document_text[:5000],  # Max 5000 chars per Snippet schema
                    source_ref=source_ref,
                    section=section,
                )
                snippets.append(snippet)

                # For longer documents, create additional snippets by paragraph
                if len(document_text) > 500:
                    paragraphs = document_text.split("\n\n")
                    for idx, para in enumerate(paragraphs[:5]):  # Max 5 paragraphs
                        if len(para) >= 50:  # Min meaningful snippet length
                            para_start = document_text.find(para)
                            para_ref = SourceRef(
                                record_id=record.record_id,
                                pmid=record.pmid,
                                doi=record.doi,
                                section=section,
                                page=None,
                                offset_start=para_start,
                                offset_end=para_start + len(para),
                            )
                            para_snippet = Snippet(
                                snippet_id=f"{record.record_id}_snip_{idx + 1}",
                                text=para[:5000],
                                source_ref=para_ref,
                                section=section,
                            )
                            snippets.append(para_snippet)

        span.set_attribute("parsed_count", len(parsed))
        span.set_attribute("snippets_count", len(snippets))
        span.set_attribute("reports_not_retrieved", reports_not_retrieved)
        span.set_attribute("fulltext_retrieved", fulltext_retrieved)
        print(f"[ParseDocs] Parsed {len(parsed)} documents, extracted {len(snippets)} snippets")
        if fulltext_retrieved > 0:
            print(f"[ParseDocs] ✓ {fulltext_retrieved} records retrieved from PMC full-text")
        if reports_not_retrieved > 0:
            print(
                f"[ParseDocs] ⚠️ {reports_not_retrieved} records without retrievable content "
                f"(counted as 'reports not retrieved' per PRISMA 2020)"
            )

        # Update PRISMA counts with reports_not_retrieved
        # Per PRISMA 2020 Flow Diagram: reports_sought = included after screening
        # Refs: https://www.prisma-statement.org/prisma-2020
        old_prisma = state.prisma_counts
        if old_prisma:
            updated_prisma = PRISMACounts(
                records_identified=old_prisma.records_identified,
                records_from_pubmed=old_prisma.records_from_pubmed,
                records_from_clinical_trials=old_prisma.records_from_clinical_trials,
                records_from_other=old_prisma.records_from_other,
                duplicates_removed=old_prisma.duplicates_removed,
                records_screened=old_prisma.records_screened,
                records_excluded=old_prisma.records_excluded,
                # PRISMA Phase 3: Reports
                reports_sought=len(included_records),
                reports_not_retrieved=reports_not_retrieved,
                reports_assessed=len(records_with_snippet),
                reports_excluded=old_prisma.reports_excluded,
                studies_included=len(records_with_snippet),
                exclusion_reasons=old_prisma.exclusion_reasons,
            )
        else:
            # Fallback if no previous PRISMA state
            updated_prisma = PRISMACounts(
                reports_sought=len(included_records),
                reports_not_retrieved=reports_not_retrieved,
                reports_assessed=len(records_with_snippet),
                studies_included=len(records_with_snippet),
            )

        # =====================================================================
        # SNIPPET VALIDATION GATE
        # Filter snippets that contain population exclusion patterns
        # This catches cases where the record passed screening but the actual
        # text contains exclusion of the PICO population (e.g., "without AF")
        # =====================================================================
        enable_snippet_gates = configurable.get("enable_evidence_gates", True)

        if enable_snippet_gates and state.pico:
            from cdr.verification.evidence_gates import PopulationMatchGate, GateResult

            pop_gate = PopulationMatchGate(strict=False)
            validated_snippets = []
            filtered_snippets = 0

            for snippet in snippets:
                # Check if snippet text contains population exclusion patterns
                gate_result = pop_gate.check_snippet(snippet, state.pico)

                if gate_result.failed:
                    filtered_snippets += 1
                    print(
                        f"[ParseDocs] Gate filtered snippet {snippet.snippet_id}: "
                        f"population excluded"
                    )
                else:
                    validated_snippets.append(snippet)

            if filtered_snippets > 0:
                print(
                    f"[ParseDocs] ⚠️ Snippet validation: {filtered_snippets} snippets "
                    f"filtered for population exclusion, {len(validated_snippets)} passed"
                )
                snippets = validated_snippets

        return {
            "parsed_documents": parsed,
            "snippets": snippets,
            "prisma_counts": updated_prisma,
        }

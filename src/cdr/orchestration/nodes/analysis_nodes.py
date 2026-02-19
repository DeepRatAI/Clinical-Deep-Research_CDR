"""
CDR Analysis Nodes

Node functions for data extraction and risk of bias assessment.
Extracted from graph.py monolith.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from cdr.core.schemas import CDRState
from cdr.observability.tracer import tracer


async def extract_data_node(state: CDRState, config: RunnableConfig) -> dict:
    """Extract structured data into StudyCards.

    Node: EXTRACT_DATA
    Input: included_records, parsed_documents
    Output: study_cards

    DESIGN: LLM provider is passed from config, not created internally.
    This ensures consistent model usage across the pipeline.
    """
    with tracer.start_span("node.extract_data") as span:
        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        from cdr.extraction.extractor import StudyCardExtractor

        # Pass LLM provider from config for consistent model usage
        extractor = StudyCardExtractor(provider=llm, model=model)
        study_cards = []
        extraction_errors: list[str] = []

        included = state.get_included_records()
        print(f"[Extract] Processing {len(included)} records")

        for record in included:
            parsed = state.parsed_documents.get(record.record_id, {})
            text = parsed.get("text", record.abstract or "")

            try:
                # extract() is sync and takes: record_id, text, title, pmid
                # CRITICAL: Use record.pmid for PubMed, record.nct_id for CT.gov
                # This preserves bibliographic traceability per NCBI/CT.gov standards
                pmid_value = record.pmid if record.pmid else None
                card = extractor.extract(
                    record_id=record.record_id,
                    text=text,
                    title=record.title,
                    pmid=pmid_value,
                )
                study_cards.append(card)
                print(f"[Extract] Card extracted for {record.record_id}")
            except Exception as e:
                # NO SILENT FAILURES: Record the error explicitly
                error_msg = f"Extraction failed for {record.record_id}: {e!s}"
                print(f"[Extract] ERROR: {error_msg}")
                extraction_errors.append(error_msg)
                span.set_attribute(f"extract_error_{record.record_id}", str(e))

        span.set_attribute("cards_extracted", len(study_cards))
        span.set_attribute("extraction_errors", len(extraction_errors))
        print(f"[Extract] Total cards: {len(study_cards)}, Errors: {len(extraction_errors)}")

        # Add errors to state for transparency
        updated_errors = list(state.errors) + extraction_errors

        return {"study_cards": study_cards, "errors": updated_errors}


async def assess_rob2_node(state: CDRState, config: RunnableConfig) -> dict:
    """Assess risk of bias using appropriate framework based on study type.

    Node: ASSESS_ROB2
    Input: study_cards, parsed_documents
    Output: rob2_results, robins_i_results

    HIGH-3 fix: Routes by StudyType - RoB2 for RCTs, ROBINS-I for observational.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-3

    Study type routing:
    - RCT, META_ANALYSIS, SYSTEMATIC_REVIEW → RoB2
    - COHORT, CASE_CONTROL, CROSS_SECTIONAL → ROBINS-I
    - Other types → RoB2 (conservative default)

    DESIGN: Failures are explicit, not silenced.
    If assessment fails for a study, the error is logged and the study
    is marked as ASSESSMENT_FAILED (not default SOME_CONCERNS).
    """
    with tracer.start_span("node.assess_rob2") as span:
        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        from cdr.rob2.assessor import RoB2Assessor
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor
        from cdr.core.schemas import (
            RoB2Result,
            RoB2DomainResult,
            ROBINSIResult,
            ROBINSIDomainResult,
        )
        from cdr.core.enums import (
            RoB2Domain,
            RoB2Judgment,
            ROBINSIDomain,
            ROBINSIJudgment,
            StudyType,
        )

        # Study types that use ROBINS-I (observational studies)
        ROBINS_I_STUDY_TYPES = {
            StudyType.COHORT,
            StudyType.CASE_CONTROL,
            StudyType.CROSS_SECTIONAL,
        }

        # Use LLM provider from config if available
        assessor = RoB2Assessor(provider=llm, model=model)
        rob2_results = []
        robins_i_results = []
        assessment_errors: list[str] = []

        # Separate studies by type for appropriate tool routing
        rct_studies = []
        observational_studies = []

        for card in state.study_cards:
            if card.study_type in ROBINS_I_STUDY_TYPES:
                observational_studies.append(card)
            else:
                rct_studies.append(card)

        print(
            f"[RiskOfBias] Routing: {len(rct_studies)} RCTs (RoB2), {len(observational_studies)} observational (ROBINS-I)"
        )
        span.set_attribute("rct_count", len(rct_studies))
        span.set_attribute("observational_count", len(observational_studies))

        # Assess RCT studies with RoB2
        for card in rct_studies:
            parsed = state.parsed_documents.get(card.record_id, {})
            text = parsed.get("text", "")

            # CLINICAL IMPROVEMENT: Use Methods section for RoB2 when available
            # RoB2 assessment quality improves dramatically with Methods text
            # vs abstract-only (which produces uniform "some_concerns")
            # Refs: Cochrane RoB2 tool documentation, PRISMA 2020
            sections = parsed.get("sections")
            if sections and isinstance(sections, dict):
                # Build RoB2-optimized text from structured sections
                rob2_parts = []
                # Methods is the MOST important section for RoB2 assessment
                if "methods" in sections:
                    rob2_parts.append(f"## Methods\n{sections['methods']}")
                # Results help assess missing data and selective reporting
                if "results" in sections:
                    rob2_parts.append(f"## Results\n{sections['results']}")
                # Abstract provides overview context
                if "abstract" in sections:
                    rob2_parts.append(f"## Abstract\n{sections['abstract']}")
                if rob2_parts:
                    text = "\n\n".join(rob2_parts)
                    print(
                        f"[RoB2] Using structured sections for {card.record_id} (Methods+Results)"
                    )

            study_info = {
                "study_type": card.study_type.value if card.study_type else None,
                "population_n": card.sample_size,
                "intervention_description": card.intervention_extracted,
                "comparator_description": card.comparator_extracted,
            }

            try:
                result = assessor.assess(card.record_id, text, study_info)
                rob2_results.append(result)
                print(f"[RoB2] Assessed {card.record_id}: {result.overall_judgment.value}")
            except Exception as e:
                error_msg = f"RoB2 assessment failed for {card.record_id}: {str(e)}"
                print(f"[RoB2] ERROR: {error_msg}")
                assessment_errors.append(error_msg)
                span.set_attribute(f"rob2_error_{card.record_id}", str(e))

                failed_domains = [
                    RoB2DomainResult(
                        domain=domain,
                        judgment=RoB2Judgment.HIGH,
                        rationale=f"ASSESSMENT FAILED: {str(e)}",
                        supporting_snippet_ids=[],
                    )
                    for domain in RoB2Domain
                ]
                failed_result = RoB2Result(
                    record_id=card.record_id,
                    domains=failed_domains,
                    overall_judgment=RoB2Judgment.HIGH,
                    overall_rationale=f"ASSESSMENT FAILED: {str(e)} - High risk assumed conservatively.",
                )
                rob2_results.append(failed_result)

        # Assess observational studies with ROBINS-I (HIGH-3 fix: proper ROBINS-I assessment)
        robins_i_assessor = ROBINSIAssessor(provider=llm, model=model)

        for card in observational_studies:
            parsed = state.parsed_documents.get(card.record_id, {})
            text = parsed.get("text", "")

            study_info = {
                "study_type": card.study_type.value if card.study_type else "observational",
                "population_n": card.sample_size,
                "intervention_description": card.intervention_extracted,
                "comparator_description": card.comparator_extracted,
            }

            try:
                result = robins_i_assessor.assess(card.record_id, text, study_info)
                robins_i_results.append(result)
                print(f"[ROBINS-I] Assessed {card.record_id}: {result.overall_judgment.value}")
            except Exception as e:
                error_msg = f"ROBINS-I assessment failed for {card.record_id}: {str(e)}"
                print(f"[ROBINS-I] ERROR: {error_msg}")
                assessment_errors.append(error_msg)
                span.set_attribute(f"robinsi_error_{card.record_id}", str(e))

                # Conservative failure: CRITICAL risk for failed observational assessment
                failed_domains = [
                    ROBINSIDomainResult(
                        domain=domain,
                        judgment=ROBINSIJudgment.CRITICAL,
                        rationale=f"ASSESSMENT FAILED: {str(e)}",
                        supporting_snippet_ids=[],
                    )
                    for domain in ROBINSIDomain
                ]
                failed_result = ROBINSIResult(
                    record_id=card.record_id,
                    domains=failed_domains,
                    overall_judgment=ROBINSIJudgment.CRITICAL,
                    overall_rationale=f"ASSESSMENT FAILED: {str(e)} - Critical risk assumed conservatively.",
                )
                robins_i_results.append(failed_result)

        span.set_attribute("rob2_count", len(rob2_results))
        span.set_attribute("robins_i_count", len(robins_i_results))
        span.set_attribute("assessment_errors", len(assessment_errors))
        print(
            f"[RiskOfBias] Total: {len(rob2_results)} RoB2, {len(robins_i_results)} ROBINS-I, {len(assessment_errors)} errors"
        )

        updated_errors = list(state.errors) + assessment_errors

        return {
            "rob2_results": rob2_results,
            "robins_i_results": robins_i_results,
            "errors": updated_errors,
        }

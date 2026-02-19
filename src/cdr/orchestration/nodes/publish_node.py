"""
CDR Publish Node

Node function for final report generation and scientific status determination.
Extracted from graph.py monolith.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from cdr.core.enums import RunStatus
from cdr.core.schemas import CDRState
from cdr.observability.tracer import tracer


async def publish_node(state: CDRState, config: RunnableConfig) -> dict:
    """Generate final report.

    Node: PUBLISH
    Input: all state
    Output: report_path

    CRITICAL: Determines final status based on scientific criteria, not just technical.

    Status determination:
    - INSUFFICIENT_EVIDENCE: No claims or no studies
    - UNPUBLISHABLE: Claims exist but lack proper supporting snippets,
                     or RoB2 invalid, or verification failed
    - COMPLETED: Technically complete AND scientifically publishable

    IMPLEMENTATION NOTE:
    This is a simplified publish implementation.
    A full implementation should generate:
    - PRISMA flow diagram (PDF/SVG)
    - Evidence tables (CSV/Excel)
    - GRADE summary of findings table
    - Full narrative report (Markdown/PDF)
    - Machine-readable JSON export
    - Citation list in standard formats

    Current implementation:
    - Generates a summary JSON file with key results
    - Sets appropriate status based on scientific criteria
    """
    import json
    import os
    from pathlib import Path

    configurable = config.get("configurable", {})
    with tracer.start_span("node.publish") as span:
        output_dir = configurable.get("output_dir", "reports")
        dod_level = configurable.get("dod_level", 1)
        run_id = state.run_id

        span.set_attribute("dod_level", dod_level)

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Determine scientific status (not just technical completion)
        # GATE FORMAL (PRISMA + GRADE): Status basado en evidencia trazable
        final_status = RunStatus.COMPLETED
        status_reason = "technically_complete"

        # Build snippet lookup for validation (used in multiple checks)
        snippet_ids = {s.snippet_id for s in state.snippets} if state.snippets else set()

        # Check for INSUFFICIENT_EVIDENCE conditions (no evidence at all)
        # Refs: PRISMA 2020 - transparency about evidence availability
        # CRITICAL: Order matters - check from earliest pipeline stage to latest
        # to ensure proper status_reason reflects the actual failure point
        # Refs: CDR_Integral_Audit_2026-01-20.md CRITICAL-1
        if not state.retrieved_records:
            # Early-exit: no records retrieved at all
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "no_records_retrieved"
        elif state.flags.get("screening_blocked_no_llm"):
            # ALTO-D fix: LLM required for screening at this DoD level
            # Refs: CDR_Integral_Audit_2026-01-20.md ALTO-D (reason_code real)
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "llm_required_for_level_2"
        elif not state.get_included_records():
            # Early-exit: all records excluded by screening
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "no_records_included_after_screening"
        elif not state.snippets:
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "no_snippets_extracted"
        elif not state.study_cards:
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "no_studies_included"
        elif not state.claims:
            final_status = RunStatus.INSUFFICIENT_EVIDENCE
            status_reason = "no_claims_generated"

        # Check for UNPUBLISHABLE conditions (claims exist but are invalid)
        # Refs: GRADE - certainty requires traceable support
        elif final_status == RunStatus.COMPLETED:
            # GATE: Verify ALL claims have valid supporting snippets
            claims_without_valid_snippets = []
            for claim in state.claims:
                valid_snippets = [sid for sid in claim.supporting_snippet_ids if sid in snippet_ids]
                if not valid_snippets:
                    claims_without_valid_snippets.append(claim.claim_id)

            if claims_without_valid_snippets:
                final_status = RunStatus.UNPUBLISHABLE
                status_reason = f"claims_without_valid_snippets: {claims_without_valid_snippets}"

            # Check if RoB2 was assessed (required for scientific validity)
            elif not state.rob2_results:
                final_status = RunStatus.UNPUBLISHABLE
                status_reason = "no_rob2_assessments"

            # Check for critical blockers from critique
            elif state.critique and state.critique.has_blockers():
                final_status = RunStatus.UNPUBLISHABLE
                status_reason = f"critique_blockers: {state.critique.blockers}"

        # DOD LEVEL GATES: Additional requirements for Research/SOTA grade
        # Refs: ADR-005, CDR_Post_ADR003_v3_PostChange_Audit_and_Actions.md
        if final_status == RunStatus.COMPLETED and dod_level >= 2:
            # Level 2 (Research-grade): Verification coverage gate
            from cdr.core.enums import VerificationStatus

            verified_count = 0
            total_claims = len(state.claims)

            if state.verification:
                for vr in state.verification:
                    if vr.overall_status in (
                        VerificationStatus.VERIFIED,
                        VerificationStatus.PARTIAL,
                    ):
                        verified_count += 1

            verification_coverage = verified_count / total_claims if total_claims > 0 else 0.0

            # Level 2 requires >=80% verification coverage
            if verification_coverage < 0.8:
                final_status = RunStatus.UNPUBLISHABLE
                status_reason = (
                    f"verification_coverage_insufficient: {verification_coverage:.1%} < 80% "
                    f"(required for DoD Level {dod_level})"
                )
                print(
                    f"[Publish] ⚠️ DoD Level {dod_level} requires >=80% verification coverage, "
                    f"got {verification_coverage:.1%}"
                )

        if final_status == RunStatus.COMPLETED and dod_level >= 3:
            # Level 3 (SOTA-grade): Requires grade_rationale populated
            claims_without_rationale = [c.claim_id for c in state.claims if not c.grade_rationale]
            if claims_without_rationale:
                final_status = RunStatus.UNPUBLISHABLE
                status_reason = (
                    f"grade_rationale_missing: {len(claims_without_rationale)} claims lack "
                    f"structured GRADE rationale (required for DoD Level {dod_level})"
                )
                print(f"[Publish] ⚠️ DoD Level {dod_level} requires grade_rationale for all claims")

        # =====================================================================
        # DOD3 GATE VALIDATION & HARD ENFORCEMENT
        # Run full DoD3 gates and generate Gate Report for audit trail.
        # CRITICAL: Apply HARD EXCLUSION so out-of-scope evidence cannot support claims.
        # This catches: PICO mismatches, comparator jumps, study type violations,
        # context purity issues, and assertion coverage gaps.
        # Refs: DoD3 Contract, PRISMA 2020, GRADE Handbook
        # =====================================================================
        gate_report_data = None
        dod3_validation_result = None
        dod3_enforcement_result = None

        if dod_level >= 2 and state.pico and state.claims:
            try:
                from cdr.verification.dod3_gates import DoD3Validator

                dod3_validator = DoD3Validator(strict=(dod_level >= 3))
                dod3_validation_result = dod3_validator.validate(
                    run_id=run_id,
                    pico=state.pico,
                    records=state.get_included_records(),
                    snippets=state.snippets or [],
                    claims=state.claims,
                )

                gate_report_data = dod3_validation_result.gate_report.to_dict()

                # P0-07: Use gate_report's status_reason as single source of truth
                gate_status_reason = dod3_validation_result.gate_report.status_reason
                gate_status_reason_code = dod3_validation_result.gate_report.status_reason_code

                # If DoD3 gates fail, determine if UNPUBLISHABLE or PARTIALLY_PUBLISHABLE
                # FIX 7: PARTIALLY_PUBLISHABLE if:
                # - Some records pass, some fail (mixed evidence quality)
                # - At least 1 claim has valid supporting evidence after enforcement
                is_unpublishable = not dod3_validation_result.passed
                if is_unpublishable:
                    # Check for partial publishability
                    excluded_record_count = len(dod3_validation_result.excluded_records)
                    total_record_count = len(state.get_included_records())
                    valid_record_count = total_record_count - excluded_record_count

                    # Partially publishable if:
                    # 1. Some records PASS (valid_record_count > 0)
                    # 2. Some records FAIL (excluded_record_count > 0)
                    # 3. At least 1 claim has surviving snippets after exclusion
                    has_mixed_evidence = valid_record_count > 0 and excluded_record_count > 0

                    if has_mixed_evidence:
                        # Check if any claims have surviving snippets
                        excluded_snippet_set = set(dod3_validation_result.excluded_snippets)
                        claims_with_valid_snippets = 0
                        for claim in state.claims:
                            surviving = [
                                s
                                for s in claim.supporting_snippet_ids
                                if s not in excluded_snippet_set
                            ]
                            if surviving:
                                claims_with_valid_snippets += 1

                        if claims_with_valid_snippets > 0 and final_status == RunStatus.COMPLETED:
                            final_status = RunStatus.PARTIALLY_PUBLISHABLE
                            # P0-07: Use gate_report status_reason with partial context
                            status_reason = (
                                f"partially_publishable: {claims_with_valid_snippets}/{len(state.claims)} claims "
                                f"have valid evidence. {gate_status_reason}"
                            )
                            print(
                                f"[Publish] ⚠️ PARTIALLY_PUBLISHABLE: "
                                f"{valid_record_count}/{total_record_count} records pass, "
                                f"{claims_with_valid_snippets} claims have valid evidence"
                            )
                        elif final_status == RunStatus.COMPLETED:
                            final_status = RunStatus.UNPUBLISHABLE
                            # P0-07: Use gate_report status_reason directly
                            status_reason = gate_status_reason
                    elif final_status == RunStatus.COMPLETED:
                        final_status = RunStatus.UNPUBLISHABLE
                        # P0-07: Use gate_report status_reason directly
                        status_reason = gate_status_reason

                    print(
                        f"[Publish] ⚠️ DoD3 gates failed: "
                        f"{len(dod3_validation_result.excluded_records)} records excluded, "
                        f"{len(dod3_validation_result.excluded_snippets)} snippets excluded"
                    )
                else:
                    # P0-07: Even on success, use gate_report status_reason
                    if not status_reason:
                        status_reason = gate_status_reason
                    print(f"[Publish] ✅ DoD3 gates passed - run is publishable")

                # =========================================================
                # HARD ENFORCEMENT: Apply exclusions to evidence chain
                # BLOCKER: If record/snippet fails P/I/C/O → cannot be Supporting Evidence
                # Refs: CDR_DOD3_b3142335 audit, DoD3 Contract
                # =========================================================
                try:
                    from cdr.verification.dod3_enforcement import DoD3Enforcer

                    # Convert gate violations to enforcement format
                    gate_violations = []
                    for v in dod3_validation_result.gate_report.blocker_violations:
                        gate_violations.append(
                            {
                                "result": "fail",
                                "record_id": v.record_id,
                                "snippet_id": v.snippet_id,
                                "pmid": v.pmid,
                                "mismatch_type": v.mismatch_type.value
                                if v.mismatch_type
                                else "unknown",
                                "message": v.message,
                                "pico_component": v.pico_component,
                                "match_score": v.match_score,
                                "gate": v.gate_name,
                            }
                        )

                    enforcer = DoD3Enforcer(
                        strict=(dod_level >= 3),
                        min_snippets_per_claim=1,
                        suppress_hypotheses_on_unpublishable=True,
                    )

                    dod3_enforcement_result = enforcer.enforce(
                        run_id=run_id,
                        pico=state.pico,
                        records=state.get_included_records(),
                        snippets=state.snippets or [],
                        claims=state.claims,
                        hypotheses=state.composed_hypotheses or [],
                        gate_violations=gate_violations,
                        is_unpublishable=is_unpublishable,
                    )

                    print(
                        f"[Publish] DoD3 Enforcement applied: "
                        f"{len(dod3_enforcement_result.excluded_records)} records excluded, "
                        f"{len(dod3_enforcement_result.excluded_snippets)} snippets excluded, "
                        f"{len(dod3_enforcement_result.orphan_claims)} claims orphaned, "
                        f"{len(dod3_enforcement_result.suppressed_hypotheses)} hypotheses suppressed"
                    )

                except Exception as e:
                    print(f"[Publish] Warning: DoD3 enforcement failed: {e}")

            except Exception as e:
                print(f"[Publish] Warning: DoD3 validation failed with error: {e}")
                span.set_attribute("dod3_validation_error", str(e))

        # Generate summary report
        # CRITICAL: Include ALL traceability fields per PRISMA 2020 / GRADE
        # Refs: ADR-005, CDR_Post_ADR003_v3_PostChange_Audit_and_Actions.md
        # Missing these fields breaks auditability and SOTA-grade compliance

        # Build RoB2 lookup by record_id for claim-level linkage
        rob2_by_record = {r.record_id: r for r in state.rob2_results} if state.rob2_results else {}

        # Build claim-to-record mapping from snippet_ids
        def get_records_from_snippets(snippet_ids: list[str]) -> list[str]:
            """Extract record_ids from snippet_ids (format: {record_id}_snip_{N})."""
            record_ids = set()
            for snip_id in snippet_ids:
                parts = snip_id.rsplit("_snip", 1)
                if parts:
                    record_ids.add(parts[0])
            return sorted(record_ids)

        # Calculate KPIs per run (required for Research/SOTA grade)
        # Refs: CDR_Post_ADR003_v3_Audit_with_Run_KPIs_and_MinEvidence_Checklist.md
        total_claims = len(state.claims)
        claims_with_snippets = sum(1 for c in state.claims if c.supporting_snippet_ids)
        snippet_coverage = claims_with_snippets / total_claims if total_claims > 0 else 0.0

        # Verification coverage
        verified_claims = 0
        unverified_claims = 0
        if state.verification:
            from cdr.core.enums import VerificationStatus

            for vr in state.verification:
                if vr.overall_status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIAL):
                    verified_claims += 1
                elif vr.overall_status == VerificationStatus.UNVERIFIABLE:
                    unverified_claims += 1
        verification_coverage = verified_claims / total_claims if total_claims > 0 else 0.0

        # Claims-with-evidence ratio (claims / records with snippets)
        records_with_snippets = (
            len({s.source_ref.record_id for s in state.snippets}) if state.snippets else 0
        )
        claims_evidence_ratio = (
            total_claims / records_with_snippets if records_with_snippets > 0 else 0.0
        )

        # Negative outcome detection
        is_negative_outcome = final_status in (
            RunStatus.INSUFFICIENT_EVIDENCE,
            RunStatus.UNPUBLISHABLE,
        )

        run_kpis = {
            "snippet_coverage": round(snippet_coverage, 3),
            "verification_coverage": round(verification_coverage, 3),
            "claims_with_evidence_ratio": round(claims_evidence_ratio, 3),
            "total_claims": total_claims,
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
            "is_negative_outcome": is_negative_outcome,
            "dod_level": configurable.get("dod_level", 1),
            # NEW: Track if synthesis used markdown fallback
            # Refs: ADR-005, CDR_Post_ADR005_Full_Audit (MEDIO)
            "used_markdown_fallback": (
                state.synthesis_result.used_markdown_fallback if state.synthesis_result else None
            ),
        }

        # =====================================================================
        # CONCLUSION DEGRADATION: Conclusions must obey UNPUBLISHABLE status
        # CRITICAL: If status == unpublishable, conclusion cannot affirm effects
        # FIX 7: Handle PARTIALLY_PUBLISHABLE with appropriate messaging
        # Refs: DoD3 Contract, PRISMA 2020 Transparency
        # =====================================================================
        final_answer = state.answer
        conclusion_degradation_result = None

        if final_status in (RunStatus.UNPUBLISHABLE, RunStatus.INSUFFICIENT_EVIDENCE):
            from cdr.verification.conclusion_enforcer import degrade_conclusion_for_unpublishable

            blocker_count = (
                len(dod3_validation_result.gate_report.blocker_violations)
                if dod3_validation_result and dod3_validation_result.gate_report
                else 0
            )

            conclusion_degradation_result = degrade_conclusion_for_unpublishable(
                original_conclusion=state.answer or "",
                status_reason=status_reason,
                gate_report=gate_report_data,
                blocker_count=blocker_count,
            )

            if conclusion_degradation_result.was_degraded:
                final_answer = conclusion_degradation_result.degraded
                print(
                    f"[Publish] Conclusion degraded for UNPUBLISHABLE status. "
                    f"Reasons: {conclusion_degradation_result.reasons[:3]}"
                )
                span.set_attribute("conclusion_degraded", True)
                span.set_attribute(
                    "degradation_reasons", str(conclusion_degradation_result.reasons)
                )

        elif final_status == RunStatus.PARTIALLY_PUBLISHABLE:
            from cdr.verification.conclusion_enforcer import degrade_conclusion_for_partial

            # Calculate valid claim count for partial messaging
            excluded_snippet_set = (
                set(dod3_validation_result.excluded_snippets) if dod3_validation_result else set()
            )

            valid_claim_count = sum(
                1
                for c in state.claims
                if any(s not in excluded_snippet_set for s in c.supporting_snippet_ids)
            )

            conclusion_degradation_result = degrade_conclusion_for_partial(
                original_conclusion=state.answer or "",
                status_reason=status_reason,
                valid_claim_count=valid_claim_count,
                total_claim_count=len(state.claims),
                gate_report=gate_report_data,
            )

            if conclusion_degradation_result.was_degraded:
                final_answer = conclusion_degradation_result.degraded
                print(
                    f"[Publish] Conclusion marked PARTIALLY_PUBLISHABLE. "
                    f"Valid claims: {valid_claim_count}/{len(state.claims)}"
                )
                span.set_attribute("conclusion_degraded", True)
                span.set_attribute(
                    "degradation_reasons", str(conclusion_degradation_result.reasons)
                )

        report_data = {
            "run_id": run_id,
            "question": state.question,
            "pico": state.pico.model_dump() if state.pico else None,
            # PRISMA-S: Include reproducible search strategy for auditability
            # Refs: PRISMA-S (BMJ 2021), CDR_Integral_Audit_2026-01-20.md HIGH-4
            "search_plan": (
                {
                    "pubmed_query": state.search_plan.pubmed_query,
                    "ct_gov_query": state.search_plan.ct_gov_query,
                    "date_range": state.search_plan.date_range,
                    "languages": state.search_plan.languages,
                    "max_results_per_source": state.search_plan.max_results_per_source,
                    "created_at": state.search_plan.created_at.isoformat()
                    if state.search_plan.created_at
                    else None,
                }
                if state.search_plan
                else None
            ),
            # HIGH-4: Track executed searches (may differ from planned due to truncation)
            # Refs: PRISMA-S (BMJ 2021), CDR_Integral_Audit_2026-01-20.md HIGH-4
            "executed_searches": [
                {
                    "database": es.database,
                    "query_planned": es.query_planned,
                    "query_executed": es.query_executed,
                    "executed_at": es.executed_at.isoformat() if es.executed_at else None,
                    "results_count": es.results_count,
                    "results_fetched": es.results_fetched,
                    "notes": es.notes,
                }
                for es in state.executed_searches
            ]
            if state.executed_searches
            else [],
            "prisma_counts": state.prisma_counts.model_dump() if state.prisma_counts else None,
            "study_count": len(state.study_cards),
            "claim_count": len(state.claims),
            "snippet_count": len(state.snippets),
            # FULL CLAIM TRACEABILITY - includes all GRADE/PRISMA required fields
            # CRITICAL: Filter out excluded snippets per DoD3 enforcement
            # Refs: DoD3 Contract, CDR_DOD3_b3142335 audit
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "claim_text": c.claim_text,
                    "certainty": c.certainty.value if c.certainty else None,
                    # HARD EXCLUSION: Only include valid snippets post-DoD3 enforcement
                    "supporting_snippet_ids": (
                        [
                            sid
                            for sid in c.supporting_snippet_ids
                            if sid in dod3_enforcement_result.valid_snippet_ids
                        ]
                        if dod3_enforcement_result
                        else c.supporting_snippet_ids
                    ),
                    # NEW: Fields required for GRADE/PRISMA auditability
                    "conflicting_snippet_ids": c.conflicting_snippet_ids,
                    "limitations": c.limitations,
                    "grade_rationale": c.grade_rationale,
                    "studies_supporting": c.studies_supporting,
                    "studies_conflicting": c.studies_conflicting,
                    # NEW: RoB2 linkage per claim (only valid records post-DoD3)
                    "linked_records": [
                        rid
                        for rid in get_records_from_snippets(c.supporting_snippet_ids)
                        if not dod3_enforcement_result
                        or rid in dod3_enforcement_result.valid_record_ids
                    ],
                    "linked_rob2": [
                        {
                            "record_id": rid,
                            "overall_judgment": rob2_by_record[rid].overall_judgment.value
                            if rid in rob2_by_record and rob2_by_record[rid].overall_judgment
                            else None,
                            "overall_rationale": rob2_by_record[rid].overall_rationale
                            if rid in rob2_by_record
                            else None,
                        }
                        for rid in get_records_from_snippets(c.supporting_snippet_ids)
                        if rid in rob2_by_record
                        and (
                            not dod3_enforcement_result
                            or rid in dod3_enforcement_result.valid_record_ids
                        )
                    ],
                    # DoD3: Mark if claim was degraded
                    "dod3_degraded": (
                        c.claim_id in [d.claim_id for d in dod3_enforcement_result.degraded_claims]
                        if dod3_enforcement_result
                        else False
                    ),
                    "dod3_orphaned": (
                        c.claim_id in dod3_enforcement_result.orphan_claims
                        if dod3_enforcement_result
                        else False
                    ),
                }
                for c in state.claims
                # Exclude orphaned claims from output
                if not dod3_enforcement_result
                or c.claim_id not in dod3_enforcement_result.orphan_claims
            ],
            # RoB2 summary for RCT studies
            "rob2_summary": [
                {
                    "record_id": r.record_id,
                    "overall_judgment": r.overall_judgment.value if r.overall_judgment else None,
                    "overall_rationale": r.overall_rationale,
                    "domains": [
                        {
                            "domain": d.domain.value,
                            "judgment": d.judgment.value,
                            "rationale": d.rationale,
                        }
                        for d in r.domains
                    ]
                    if r.domains
                    else [],
                }
                for r in state.rob2_results
            ]
            if state.rob2_results
            else [],
            # HIGH-3: ROBINS-I summary for observational studies
            # Refs: CDR_Integral_Audit_2026-01-20.md HIGH-3
            "robins_i_summary": [
                {
                    "record_id": r.record_id,
                    "overall_judgment": r.overall_judgment.value if r.overall_judgment else None,
                    "overall_rationale": r.overall_rationale,
                    "domains": [
                        {
                            "domain": d.domain.value,
                            "judgment": d.judgment.value,
                            "rationale": d.rationale,
                        }
                        for d in r.domains
                    ]
                    if r.domains
                    else [],
                }
                for r in state.robins_i_results
            ]
            if state.robins_i_results
            else [],
            # Verification results
            "verification_summary": [
                {
                    "claim_id": vr.claim_id,
                    "overall_status": vr.overall_status.value if vr.overall_status else None,
                    "overall_confidence": vr.overall_confidence,
                    "checks_count": len(vr.checks) if vr.checks else 0,
                }
                for vr in state.verification
            ]
            if state.verification
            else [],
            # NEW: Run KPIs (required for Research/SOTA grade assessment)
            "run_kpis": run_kpis,
            # HIGH-1: Compositional inference hypotheses (DoD Level 3 only)
            # Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1
            "composed_hypotheses": state.composed_hypotheses if state.composed_hypotheses else [],
            "critique_findings": len(state.critique.findings) if state.critique else 0,
            "critique_blockers": state.critique.blockers if state.critique else [],
            # DOD3 GATE REPORT - Full audit trail when gates run
            # Includes: PICO match results, study type enforcement, context purity,
            # assertion coverage, and all blocker violations
            # Refs: DoD3 Contract, PRISMA 2020 Transparency
            "gate_report": gate_report_data,
            # DOD3 ENFORCEMENT RESULTS - What was actually excluded/degraded
            # CRITICAL: This closes the loop - violations → exclusions → degradations
            "dod3_enforcement": (
                dod3_enforcement_result.to_dict() if dod3_enforcement_result else None
            ),
            "dod3_excluded_records": (
                [
                    {
                        "record_id": e.evidence_id,
                        "pmid": e.pmid,
                        "reason": e.reason.value,
                        "detail": e.detail,
                    }
                    for e in dod3_enforcement_result.excluded_records
                ]
                if dod3_enforcement_result
                else (dod3_validation_result.excluded_records if dod3_validation_result else [])
            ),
            "dod3_excluded_snippets": (
                [
                    {
                        "snippet_id": e.evidence_id,
                        "pmid": e.pmid,
                        "reason": e.reason.value,
                        "detail": e.detail,
                    }
                    for e in dod3_enforcement_result.excluded_snippets
                ]
                if dod3_enforcement_result
                else (dod3_validation_result.excluded_snippets if dod3_validation_result else [])
            ),
            "dod3_degraded_claims": (
                [d.to_dict() for d in dod3_enforcement_result.degraded_claims]
                if dod3_enforcement_result
                else (dod3_validation_result.degraded_claims if dod3_validation_result else [])
            ),
            "dod3_suppressed_hypotheses": (
                [h.to_dict() for h in dod3_enforcement_result.suppressed_hypotheses]
                if dod3_enforcement_result
                else []
            ),
            "errors": state.errors,
            "status": final_status.value,
            "status_reason": status_reason,
            # CLINICAL DISCLAIMER — always present in every report
            "disclaimer": (
                "\u26a0\ufe0f This report is machine-generated by CDR (Clinical Deep Research). "
                "It is NOT medical advice and should NOT be used for clinical decision-making. "
                "All findings require independent verification by qualified professionals. "
                "See DISCLAIMER.md for full terms."
            ),
            # CRITICAL: Include answer with degradation applied for UNPUBLISHABLE
            "answer": final_answer,
            "answer_original": state.answer
            if conclusion_degradation_result and conclusion_degradation_result.was_degraded
            else None,
            "conclusion_degraded": conclusion_degradation_result.was_degraded
            if conclusion_degradation_result
            else False,
            "conclusion_degradation_reasons": conclusion_degradation_result.reasons
            if conclusion_degradation_result
            else [],
        }

        # =============================================================
        # EVALUATION INTEGRATION: Generate EvaluationReport per run
        # Refs: CDR SOTA requirements, DoD Level 2/3 metrics
        # =============================================================
        from cdr.evaluation.metrics import evaluate_cdr_output
        from cdr.composition.schemas import ComposedHypothesis
        from cdr.core.schemas import VerificationResult

        # Build verification results dict for evaluator
        verification_dict: dict[str, VerificationResult] = {}
        if state.verification:
            for vr in state.verification:
                verification_dict[vr.claim_id] = vr

        # Parse composed hypotheses if they exist
        composed_list: list[ComposedHypothesis] = []
        if state.composed_hypotheses:
            for h_dict in state.composed_hypotheses:
                try:
                    composed_list.append(ComposedHypothesis.model_validate(h_dict))
                except Exception:
                    pass  # Skip invalid entries

        evaluation_report = evaluate_cdr_output(
            run_id=run_id,
            claims=state.claims,
            snippets=state.snippets or [],
            hypotheses=composed_list,
            verification_results=verification_dict,
            dod_level=dod_level,
        )

        # Add evaluation report to output
        report_data["evaluation"] = evaluation_report.to_dict()

        print(
            f"[Publish] Evaluation: {evaluation_report.summary} "
            f"(overall_pass={evaluation_report.overall_pass})"
        )

        report_file = os.path.join(output_dir, f"cdr_report_{run_id[:8]}.json")

        try:
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"[Publish] Report saved to {report_file}")
            print(f"[Publish] Status: {final_status.value} ({status_reason})")
            span.set_attribute("report_path", report_file)
            span.set_attribute("final_status", final_status.value)
            span.set_attribute("status_reason", status_reason)
            report_data["file_path"] = report_file
        except Exception as e:
            print(f"[Publish] Error saving report: {e}")
            span.set_attribute("publish_error", str(e))

        return {
            "report": report_data,
            "status": final_status,
            # CRITICAL: Return degraded answer so state.answer reflects UNPUBLISHABLE status
            "answer": final_answer,
        }

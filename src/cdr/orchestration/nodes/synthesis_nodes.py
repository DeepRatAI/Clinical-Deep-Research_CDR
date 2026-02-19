"""
CDR Synthesis Nodes

Node functions for evidence synthesis, critique, verification, and composition.
Extracted from graph.py monolith.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from cdr.core.schemas import CDRState
from cdr.observability.tracer import tracer
from cdr.observability.metrics import metrics


async def synthesize_node(state: CDRState, config: RunnableConfig) -> dict:
    """Synthesize evidence across studies.

    Node: SYNTHESIZE
    Input: study_cards, rob2_results, robins_i_results
    Output: synthesis_result, claims

    DoD Level Gates:
    - Level 2+: Requires JSON structured outputs (no Markdown fallback penalty-free)
    - Level 3: Requires grade_rationale populated for each claim
    """
    with tracer.start_span("node.synthesize") as span:
        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")
        dod_level = configurable.get("dod_level", 1)

        span.set_attribute("dod_level", dod_level)
        print(
            f"[Synthesize] Processing {len(state.study_cards)} studies, "
            f"{len(state.rob2_results)} RoB2 results (DoD Level {dod_level})"
        )

        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        if not llm:
            return {"errors": [*state.errors, "No LLM provider configured for synthesis"]}

        # GATE FORMAL (PRISMA + GRADE): Build set of valid snippet IDs
        # Claims can ONLY reference snippets that exist in state.snippets
        # Refs: PRISMA 2020 (trazabilidad), GRADE handbook (certeza basada en evidencia)
        valid_snippet_ids: set[str] = set()
        if state.snippets:
            valid_snippet_ids = {s.snippet_id for s in state.snippets}

        span.set_attribute("valid_snippet_count", len(valid_snippet_ids))
        print(f"[Synthesize] Available snippets for claims: {len(valid_snippet_ids)}")

        synthesizer = EvidenceSynthesizer(llm, model)

        # synthesize() is synchronous now
        # CRITICAL: Pass valid_snippet_ids for early filtering
        # This prevents creating claims with invalid snippets that would be rejected
        # by the gate, reducing false negatives from pipeline timing issues
        # Refs: ADR-003 post-audit, PRISMA 2020 traceability
        result = synthesizer.synthesize(
            state.study_cards,
            state.rob2_results,
            state.question,
            valid_snippet_ids=valid_snippet_ids,
        )

        # GATE FORMAL: Filter claims to only those with valid snippet support
        # A claim without traceable evidence is not a scientific claim
        validated_claims = []
        rejected_claims = []

        for claim in result.claims:
            # Check if ALL supporting snippets exist in state.snippets
            claim_snippets = set(claim.supporting_snippet_ids)
            valid_support = claim_snippets & valid_snippet_ids

            if valid_support:
                # Claim has at least one valid snippet - accept it
                # CRITICAL: EvidenceClaim has frozen=True, use model_copy()
                # Refs: Pydantic v2 immutability, PRISMA 2020 traceability
                validated_claim = claim.model_copy(
                    update={"supporting_snippet_ids": list(valid_support)}
                )
                validated_claims.append(validated_claim)
            else:
                # Claim has NO valid snippets - reject per PRISMA/GRADE
                rejected_claims.append(claim.claim_id)

        if rejected_claims:
            print(
                f"[Synthesize] GATE: Rejected {len(rejected_claims)} claims without valid snippets"
            )
            span.set_attribute("rejected_claims", rejected_claims)

        print(
            f"[Synthesize] Generated {len(validated_claims)} validated claims (of {len(result.claims)} total)"
        )
        span.set_attribute("claim_count", len(validated_claims))
        span.set_attribute("claim_count_raw", len(result.claims))
        metrics.counter("cdr.claims.synthesized", len(validated_claims))

        # GRADE DOWNGRADE: Check for RoB2 assessment failures
        # Per GRADE handbook: if RoB2 couldn't be assessed, downgrade certainty
        # Refs: GRADE handbook section 5.2, Cochrane RoB2 tool
        rob2_failures = [
            r.record_id
            for r in state.rob2_results
            if "ASSESSMENT FAILED" in (r.overall_rationale or "")
        ]

        if rob2_failures:
            print(f"[Synthesize] RoB2 failures detected for: {rob2_failures}")
            span.set_attribute("rob2_failures", rob2_failures)

            # Downgrade claims if they rely on studies with RoB2 failures
            from cdr.core.enums import GRADECertainty

            downgraded_claims = []
            for claim in validated_claims:
                # Check if claim's snippets are from failed RoB2 studies
                # Get record_ids from snippet_ids (format: record_id_snip_N or similar)
                claim_records = set()
                for snip_id in claim.supporting_snippet_ids:
                    # Extract record_id from snippet_id
                    parts = snip_id.rsplit("_snip", 1)
                    if parts:
                        claim_records.add(parts[0])

                affected_by_failure = claim_records & set(rob2_failures)

                if affected_by_failure:
                    # Downgrade certainty and add limitation
                    # CRITICAL: Per GRADE handbook section 5.2, RoB2 failures
                    # must explicitly downgrade certainty with documented rationale
                    current_certainty = claim.certainty
                    downgrade_map = {
                        GRADECertainty.HIGH: GRADECertainty.MODERATE,
                        GRADECertainty.MODERATE: GRADECertainty.LOW,
                        GRADECertainty.LOW: GRADECertainty.VERY_LOW,
                        GRADECertainty.VERY_LOW: GRADECertainty.VERY_LOW,
                    }
                    new_certainty = downgrade_map.get(current_certainty, GRADECertainty.VERY_LOW)

                    # Build standardized GRADE rationale text
                    # Refs: GRADE handbook section 5.2, Cochrane RoB2 tool
                    grade_downgrade_text = (
                        f"GRADE DOWNGRADE (Risk of Bias): RoB2 assessment failed for "
                        f"studies [{', '.join(sorted(affected_by_failure))}]. "
                        f"Per GRADE handbook section 5.2: unable to assess risk of bias "
                        f"leads to mandatory downgrade. Certainty reduced from "
                        f"{current_certainty.value} to {new_certainty.value}."
                    )

                    new_limitations = [*claim.limitations, grade_downgrade_text]

                    # Build structured GRADE rationale per ADR-004
                    # Keys: risk_of_bias, inconsistency, indirectness, imprecision, publication_bias
                    new_grade_rationale = {
                        **claim.grade_rationale,
                        "risk_of_bias": (
                            f"Downgraded: RoB2 assessment failed for studies "
                            f"[{', '.join(sorted(affected_by_failure))}]. "
                            f"Per GRADE handbook section 5.2."
                        ),
                    }

                    downgraded_claim = claim.model_copy(
                        update={
                            "certainty": new_certainty,
                            "limitations": new_limitations,
                            "grade_rationale": new_grade_rationale,
                        }
                    )
                    downgraded_claims.append(downgraded_claim)
                else:
                    downgraded_claims.append(claim)

            validated_claims = downgraded_claims
            print(
                f"[Synthesize] Downgraded {len([c for c in validated_claims if 'RoB2 assessment failed' in str(c.limitations)])} claims due to RoB2 failures"
            )

        # CRITICAL: Populate answer field per CDRState contract
        # This ensures downstream consumers have a narrative answer
        # Refs: PRISMA 2020 (narrative results)
        answer = result.overall_narrative if result.overall_narrative else None
        if not answer and validated_claims:
            # Fallback: construct answer from validated claims only
            claim_texts = [c.claim_text for c in validated_claims[:5]]  # Top 5 claims
            answer = "Based on the evidence synthesis: " + " ".join(claim_texts)

        # =====================================================================
        # ASSERTION GATE: Validate conclusion semantic coherence
        # Refs: PRISMA 2020, GRADE handbook, CDR DoD P3 (no invalid leaps)
        # CRITICAL: Block conclusions that make claims not supported by evidence
        # =====================================================================
        assertion_gate_result = None
        if answer and state.pico and validated_claims:
            from cdr.verification.assertion_gate import AssertionGate

            # Use non-strict mode: degrade assertions instead of blocking
            gate = AssertionGate(strict_mode=False, fail_on_invalid_leap=True)
            assertion_gate_result = gate.validate(
                conclusion_text=answer,
                claims=validated_claims,
                pico=state.pico,
                snippets=state.snippets,
            )

            # Log gate result
            for line in assertion_gate_result.audit_trail:
                print(f"[Synthesize] {line}")

            # If there are violations (even degraded), add warnings to answer
            if assertion_gate_result.violations:
                # Append caveats to the answer
                caveats = []
                for v in assertion_gate_result.violations:
                    if v.violation_type.value == "invalid_leap":
                        caveats.append(
                            f"⚠️ CAVEAT: Evidence is primarily comparative "
                            f"({assertion_gate_result.audit_trail[2] if len(assertion_gate_result.audit_trail) > 2 else 'head-to-head'}). "
                            f"Absolute efficacy claims require placebo-controlled evidence."
                        )
                        break  # One caveat is enough
                    elif v.violation_type.value == "comparator_mismatch":
                        caveats.append(
                            f"⚠️ CAVEAT: PICO comparator may not match the evidence comparator. "
                            f"Review evidence context."
                        )
                        break

                if caveats:
                    answer = answer + "\n\n" + "\n".join(caveats)
                    span.set_attribute("assertion_gate_caveats_added", len(caveats))

            span.set_attribute("assertion_gate_passed", assertion_gate_result.passed)
            span.set_attribute("assertion_gate_violations", len(assertion_gate_result.violations))

        # =====================================================================
        # EARLY GATE: Snippets exist but no claims generated
        # Refs: ADR-005, PRISMA 2020 (transparency about evidence gaps)
        # CRITICAL: This is different from "no snippets" - evidence exists but
        # synthesis couldn't extract structured claims. This must be flagged.
        # =====================================================================
        if state.snippets and not validated_claims:
            print(
                f"[Synthesize] GATE: {len(state.snippets)} snippets exist but 0 claims generated. "
                f"Evidence exists but synthesis failed to extract claims."
            )
            span.set_attribute("gate_blocked", True)
            span.set_attribute("gate_block_reason", "NO_CLAIMS_WITH_EVIDENCE")
            span.set_attribute("snippet_count", len(state.snippets))

            return {
                "synthesis_result": result,
                "claims": [],
                "answer": None,
                "errors": [
                    *state.errors,
                    f"INSUFFICIENT_EVIDENCE: {len(state.snippets)} snippets extracted but "
                    f"synthesis produced 0 valid claims. Reason: NO_CLAIMS_WITH_EVIDENCE. "
                    f"This indicates evidence quality issues or synthesis failure.",
                ],
            }

        # =====================================================================
        # DOD LEVEL EARLY GATES (synthesize_node)
        # Refs: ADR-005 Post-Change Audit, CDR_Post_ADR005_Full_Audit
        # CRITICAL: Block early to avoid wasting compute on invalid syntheses
        # =====================================================================

        synthesis_metadata = {
            "blocked_by_dod": False,
            "dod_level": dod_level,
            "reason_code": None,
            "used_markdown_fallback": result.used_markdown_fallback,
        }

        # GATE: Level 2+ requires JSON structured outputs
        # Markdown fallback uses heuristic parsing which is not reproducible
        # Refs: PRISMA 2020 (reproducibility), GRADE handbook
        if dod_level >= 2 and result.used_markdown_fallback:
            print(
                f"[Synthesize] DOD_LEVEL_2_BLOCKED: Markdown fallback not allowed "
                f"for Research-grade (Level {dod_level})"
            )
            span.set_attribute("dod_blocked", True)
            span.set_attribute("dod_block_reason", "MARKDOWN_FALLBACK_NOT_ALLOWED")

            synthesis_metadata["blocked_by_dod"] = True
            synthesis_metadata["reason_code"] = "MARKDOWN_FALLBACK_NOT_ALLOWED"

            return {
                "synthesis_result": result,
                "claims": [],  # Block claims - cannot use heuristic parsing
                "answer": None,
                "errors": [
                    *state.errors,
                    f"DOD_LEVEL_{dod_level}_JSON_REQUIRED: Markdown fallback not allowed. "
                    f"LLM must return valid JSON for Research-grade synthesis.",
                ],
            }

        # GATE: Level 3 requires grade_rationale COMPLETE per GRADE domain
        # Per GRADE handbook: ALL 5 domains must be explicitly justified
        # Refs: ADR-005, GRADE handbook section 5.2, CDR_Post_ADR005_Full_Audit
        # CRITICAL: Not just "exists" but contains all 5 GRADE domains or "not_applicable"
        GRADE_REQUIRED_DOMAINS = frozenset(
            [
                "risk_of_bias",
                "inconsistency",
                "indirectness",
                "imprecision",
                "publication_bias",
            ]
        )

        if dod_level >= 3 and validated_claims:
            claims_with_incomplete_rationale = []
            claims_missing_domains = {}

            for claim in validated_claims:
                if not claim.grade_rationale:
                    # No rationale at all
                    claims_with_incomplete_rationale.append(claim)
                    claims_missing_domains[claim.claim_id] = list(GRADE_REQUIRED_DOMAINS)
                else:
                    # Check for missing domains
                    present_domains = set(claim.grade_rationale.keys())
                    missing_domains = GRADE_REQUIRED_DOMAINS - present_domains

                    # Check for empty values (must have content or "not_applicable")
                    empty_domains = [
                        d
                        for d in present_domains
                        if d in GRADE_REQUIRED_DOMAINS and not claim.grade_rationale.get(d)
                    ]

                    all_missing = missing_domains | set(empty_domains)

                    if all_missing:
                        claims_with_incomplete_rationale.append(claim)
                        claims_missing_domains[claim.claim_id] = sorted(all_missing)

            if claims_with_incomplete_rationale:
                missing_ids = [c.claim_id for c in claims_with_incomplete_rationale]
                print(
                    f"[Synthesize] DOD_LEVEL_3_BLOCKED: {len(claims_with_incomplete_rationale)} claims "
                    f"have incomplete grade_rationale: {claims_missing_domains}"
                )
                span.set_attribute("dod_blocked", True)
                span.set_attribute("dod_block_reason", "GRADE_RATIONALE_INCOMPLETE")
                span.set_attribute("claims_blocked", missing_ids)
                span.set_attribute("claims_missing_domains", str(claims_missing_domains))

                synthesis_metadata["blocked_by_dod"] = True
                synthesis_metadata["reason_code"] = "GRADE_RATIONALE_INCOMPLETE"
                synthesis_metadata["claims_blocked"] = missing_ids
                synthesis_metadata["claims_missing_domains"] = claims_missing_domains

                return {
                    "synthesis_result": result,
                    "claims": [],  # Block ALL claims - SOTA requires complete rationale
                    "answer": None,
                    "errors": [
                        *state.errors,
                        f"DOD_LEVEL_3_GRADE_RATIONALE_INCOMPLETE: {len(claims_with_incomplete_rationale)} claims "
                        f"have incomplete grade_rationale. Required domains: {sorted(GRADE_REQUIRED_DOMAINS)}. "
                        f"Missing: {claims_missing_domains}",
                    ],
                }

        # Normal path - no DoD blocks
        span.set_attribute("dod_blocked", False)

        return {
            "synthesis_result": result,
            "claims": validated_claims,  # GATE: Only claims with valid snippet support
            "answer": answer,
        }


async def critique_node(state: CDRState, config: RunnableConfig) -> dict:
    """Run skeptic agent to critique claims.

    Node: CRITIQUE
    Input: synthesis_result
    Output: critique

    If no claims available, returns empty Critique (not None).
    Failures are explicit, not silenced.
    """
    with tracer.start_span("node.critique") as span:
        from cdr.core.schemas import Critique

        # If no claims to critique, return explicit empty critique
        if not state.claims:
            print("[Critique] No claims to critique - returning empty critique")
            empty_critique = Critique(
                findings=[],
                blockers=["No claims generated from synthesis"],
                recommendations=["Review synthesis step - no evidence claims produced"],
                overall_assessment="Critique skipped: no claims to evaluate",
            )
            return {"critique": empty_critique}

        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        if not llm:
            print("[Critique] No LLM provider configured")
            no_llm_critique = Critique(
                findings=[],
                blockers=["No LLM provider configured for critique"],
                recommendations=["Configure LLM provider in runner"],
                overall_assessment="Critique failed: no LLM provider",
            )
            return {"critique": no_llm_critique}

        from cdr.skeptic.skeptic_agent import SkepticAgent

        skeptic = SkepticAgent(llm, model)

        # synthesis_result is required for critique
        if not state.synthesis_result:
            print("[Critique] No synthesis_result available - cannot critique")
            no_synthesis_critique = Critique(
                findings=[],
                blockers=["No synthesis result available for critique"],
                recommendations=["Ensure synthesis step completes before critique"],
                overall_assessment="Critique failed: synthesis_result is None",
            )
            return {"critique": no_synthesis_critique}

        # critique() is sync and returns Critique
        result = skeptic.critique(
            state.synthesis_result,
            state.question,
        )

        print(
            f"[Critique] Generated {len(result.findings)} findings, {len(result.blockers)} blockers"
        )
        span.set_attribute("finding_count", len(result.findings))
        span.set_attribute("blocker_count", len(result.blockers))

        return {"critique": result}


async def verify_node(state: CDRState, config: RunnableConfig) -> dict:
    """Verify claims against source snippets.

    Node: VERIFY
    Input: claims, snippets, parsed_documents
    Output: verification

    IMPLEMENTATION:
    Uses Verifier.verify_all_claims to perform LLM-based entailment checking
    between claims and their supporting snippets.

    Verification is a GATE: claims without verified support should be flagged.
    """
    with tracer.start_span("node.verify") as span:
        configurable = config.get("configurable", {})
        llm = configurable.get("llm_provider")
        model = configurable.get("model", "gpt-4o")

        from cdr.core.enums import VerificationStatus
        from cdr.core.schemas import Snippet, VerificationResult
        from cdr.verification.verifier import Verifier

        # If no claims, return empty verification result
        if not state.claims:
            print("[Verify] No claims to verify")
            span.set_attribute("skipped_reason", "no_claims")
            return {"verification": []}

        # If no LLM, mark all as UNVERIFIABLE (explicit failure, not silent pass)
        if not llm:
            print("[Verify] No LLM provider - marking all claims as UNVERIFIABLE")
            span.set_attribute("skipped_reason", "no_llm")
            verification_results = []
            for claim in state.claims:
                placeholder = VerificationResult(
                    claim_id=claim.claim_id,
                    overall_status=VerificationStatus.UNVERIFIABLE,
                    overall_confidence=0.0,
                    checks=[],
                )
                verification_results.append(placeholder)
            return {"verification": verification_results}

        # Build snippets dict from state
        snippets_dict: dict[str, Snippet] = {}
        if state.snippets:
            for snippet in state.snippets:
                snippets_dict[snippet.snippet_id] = snippet

        # Build source_texts dict from parsed documents
        # CRITICAL: parsed_documents is dict[str, dict] per CDRState schema
        # Each value has 'text', 'source', 'title' keys (set in parse_documents_node)
        source_texts: dict[str, str] = {}
        if state.parsed_documents:
            for record_id, doc_data in state.parsed_documents.items():
                # Use the 'text' field that parse_documents_node sets
                text = doc_data.get("text", "")
                if text:
                    source_texts[record_id] = text

        # DEBUG: Log source_texts availability
        print(f"[Verify] source_texts available for {len(source_texts)} records")
        if source_texts:
            sample_id = list(source_texts.keys())[0]
            sample_len = len(source_texts[sample_id])
            print(f"[Verify] Sample source_text: {sample_id} ({sample_len} chars)")

        # If no snippets, verification cannot proceed
        if not snippets_dict:
            print("[Verify] No snippets available - cannot verify claims")
            span.set_attribute("skipped_reason", "no_snippets")
            verification_results = []
            for claim in state.claims:
                placeholder = VerificationResult(
                    claim_id=claim.claim_id,
                    overall_status=VerificationStatus.UNVERIFIABLE,
                    overall_confidence=0.0,
                    checks=[],
                )
                verification_results.append(placeholder)
            return {"verification": verification_results}

        # Perform real verification
        print(
            f"[Verify] Verifying {len(state.claims)} claims against {len(snippets_dict)} snippets"
        )

        verifier = Verifier(llm, model)
        results_dict = verifier.verify_all_claims(
            claims=state.claims,
            snippets=snippets_dict,
            source_texts=source_texts,
        )

        # Convert dict to list for state
        verification_results = list(results_dict.values())

        # Log summary statistics
        verified_count = sum(
            1 for r in verification_results if r.overall_status == VerificationStatus.VERIFIED
        )
        partial_count = sum(
            1 for r in verification_results if r.overall_status == VerificationStatus.PARTIAL
        )
        contradicted_count = sum(
            1 for r in verification_results if r.overall_status == VerificationStatus.CONTRADICTED
        )

        span.set_attribute("claims_verified", verified_count)
        span.set_attribute("claims_partial", partial_count)
        span.set_attribute("claims_contradicted", contradicted_count)
        span.set_attribute("claims_total", len(verification_results))

        print(
            f"[Verify] Results: {verified_count} verified, {partial_count} partial, {contradicted_count} contradicted"
        )

        return {"verification": verification_results}


async def compose_node(state: CDRState, config: RunnableConfig) -> dict:
    """Generate compositional inferences from verified claims.

    Node: COMPOSE
    Input: claims, verification, pico
    Output: composed_hypotheses

    IMPLEMENTATION (HIGH-1):
    Uses CompositionEngine to detect mechanistic relations between claims
    and propose novel hypotheses from composition (A+B⇒C).

    This node is CONDITIONAL: only runs for DoD Level 3 (full review).
    Levels 1-2 skip this step and proceed directly to PUBLISH.

    Composed hypotheses are marked as DERIVED, not as primary findings.
    They require additional validation and should be flagged for user review.

    Refs: Bradford Hill criteria for causality assessment,
          Cochrane Handbook Section 12.2.2 (indirect comparisons)
    """
    with tracer.start_span("node.compose") as span:
        configurable = config.get("configurable", {})
        dod_level = configurable.get("dod_level", 1)
        llm = configurable.get("llm_provider")

        # HIGH-1: Compositional inference only for Level 3
        if dod_level < 3:
            print(f"[Compose] Skipping for DoD level {dod_level} (requires Level 3)")
            span.set_attribute("skipped_reason", f"dod_level_{dod_level}")
            return {"composed_hypotheses": []}

        # Must have claims and verification to compose
        if not state.claims:
            print("[Compose] No claims available - cannot compose")
            span.set_attribute("skipped_reason", "no_claims")
            return {"composed_hypotheses": []}

        if not state.verification:
            print("[Compose] No verification results - cannot compose")
            span.set_attribute("skipped_reason", "no_verification")
            return {"composed_hypotheses": []}

        # Filter to verified claims only (we don't compose from unverified claims)
        from cdr.core.enums import VerificationStatus

        verified_claim_ids = {
            v.claim_id
            for v in state.verification
            if v.overall_status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIAL)
        }

        verified_claims = [c for c in state.claims if c.claim_id in verified_claim_ids]

        if len(verified_claims) < 2:
            print(
                f"[Compose] Insufficient verified claims ({len(verified_claims)}) - need at least 2"
            )
            span.set_attribute("skipped_reason", "insufficient_claims")
            span.set_attribute("verified_claims_count", len(verified_claims))
            return {"composed_hypotheses": []}

        # If no LLM, cannot perform composition
        if not llm:
            print("[Compose] No LLM provider - cannot perform compositional inference")
            span.set_attribute("skipped_reason", "no_llm")
            return {"composed_hypotheses": []}

        print(f"[Compose] Composing from {len(verified_claims)} verified claims")

        # Run CompositionEngine
        from cdr.composition import CompositionEngine

        engine = CompositionEngine(provider=llm)
        composed = engine.run(verified_claims, pico=state.pico)

        # Convert to dicts for state storage
        composed_dicts = [h.model_dump(mode="json") for h in composed]

        span.set_attribute("hypotheses_generated", len(composed_dicts))
        print(f"[Compose] Generated {len(composed_dicts)} composed hypotheses")

        return {"composed_hypotheses": composed_dicts}

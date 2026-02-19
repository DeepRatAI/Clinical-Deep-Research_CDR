"""
CDR Synthesis Layer

Evidence synthesis with GRADE assessment and meta-narrative generation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from cdr.core.enums import GRADECertainty, OutcomeMeasureType, Section, TherapeuticContext
from cdr.core.schemas import (
    EvidenceClaim,
    OutcomeMeasure,
    RoB2Result,
    StudyCard,
    SynthesisResult,  # Import from schemas
)
from cdr.observability.tracer import tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a systematic review synthesis expert.

Your task is to synthesize evidence from multiple studies into coherent claims
with GRADE certainty assessments.

For each outcome or theme, you must:
1. Aggregate findings across studies
2. Assess heterogeneity (I² if quantitative)
3. Consider risk of bias from RoB2 assessments
4. Evaluate directness, precision, and publication bias
5. Assign GRADE certainty: high, moderate, low, very_low
6. CLASSIFY THERAPEUTIC CONTEXT for each claim

GRADE Downgrading Factors:
- Risk of bias: High/some concerns in >50% studies → downgrade
- Inconsistency: I² > 50% or conflicting results → downgrade
- Indirectness: Population/intervention differences → downgrade
- Imprecision: Wide CIs, small sample sizes → downgrade
- Publication bias: Funnel plot asymmetry, missing studies → downgrade

EPISTEMIC LANGUAGE GUIDELINES (Based on Cochrane/GRADE standards):
Use language that matches the certainty level:
- HIGH certainty: "X reduces/increases Y", "X is effective for Y"
- MODERATE certainty: "X probably reduces Y", "X likely improves Y"
- LOW certainty: "X may reduce Y", "X is associated with Y", "evidence suggests X"
- VERY_LOW certainty: "X might reduce Y", "uncertain whether X affects Y", "very limited evidence"

NEVER use definitive language ("X is effective") for low/very_low certainty claims.
ALWAYS qualify claims appropriately based on the evidence strength.

THERAPEUTIC CONTEXT CLASSIFICATION (CRITICAL):
Each claim MUST be tagged with its therapeutic context. Never mix contexts in one claim.

Available contexts:
- "monotherapy": Drug alone vs placebo/no treatment
- "monotherapy_vs_active": Drug A vs Drug B (head-to-head)
- "add_on": Drug added to existing therapy
- "aspirin_monotherapy": Aspirin alone context
- "aspirin_plus_anticoagulant": Aspirin + anticoagulant context  
- "doac_vs_aspirin": DOAC compared to aspirin (head-to-head)
- "doac_vs_warfarin": DOAC compared to warfarin
- "head_to_head": Active comparator (superiority/non-inferiority)
- "unclassified": Could not determine

CRITICAL COHERENCE RULES:
1. If all evidence is COMPARATIVE (Drug A vs Drug B), claims MUST reflect this comparison.
   - WRONG: "Aspirin reduces stroke" (implies vs placebo)
   - RIGHT: "In head-to-head trials, DOACs showed lower stroke rates than aspirin"
2. NEVER claim absolute efficacy when evidence is only comparative.
3. If combining contexts, make it EXPLICIT with limitations.

CRITICAL: YOU MUST OUTPUT ONLY VALID JSON. NO MARKDOWN. NO TEXT BEFORE OR AFTER.
Start your response with { and end with }.

IMPORTANT: Use these EXACT field names (they are mandatory):
- claim_text (NOT statement)
- supporting_snippet_ids (NOT supporting_studies)
- therapeutic_context (REQUIRED - one of the contexts above)
- certainty must be lowercase: high, moderate, low, very_low

Output format:
{
    "claims": [
        {
            "claim_id": "claim_001",
            "claim_text": "Epistemic claim matching certainty level (min 20 chars)",
            "certainty": "moderate",
            "therapeutic_context": "doac_vs_aspirin",
            "supporting_snippet_ids": ["record_001_snip_0", "record_002_snip_0"],
            "limitations": ["Known limitation 1"],
            "studies_supporting": 3,
            "grade_rationale": {
                "risk_of_bias": "some concerns in 2/5 studies",
                "inconsistency": "I² = 35%, acceptable",
                "indirectness": "direct - populations match",
                "imprecision": "adequate sample size",
                "publication_bias": "unlikely - comprehensive search"
            }
        }
    ],
    "heterogeneity_assessment": {
        "clinical": "description of clinical heterogeneity",
        "methodological": "description of methodological heterogeneity",
        "statistical": "I² value if applicable"
    },
    "overall_synthesis": "Narrative summary of synthesized evidence"
}

Rules:
- Every claim MUST have supporting_snippet_ids (list of snippet IDs, min 1)
- Every claim MUST have therapeutic_context (NEVER leave it out)
- claim_text MUST be at least 20 characters
- certainty MUST be lowercase: high, moderate, low, very_low
- Certainty MUST be justified by GRADE rationale
- Be conservative - when uncertain, downgrade
- Highlight conflicts between studies explicitly
- USE EPISTEMIC LANGUAGE MATCHING THE CERTAINTY LEVEL
- DO NOT MIX THERAPEUTIC CONTEXTS in a single claim

REMEMBER: Output ONLY valid JSON, nothing else."""

META_ANALYSIS_PROMPT = """You are a meta-analysis assistant.

Given the following study outcomes, calculate pooled estimates where appropriate.

For continuous outcomes: weighted mean difference or standardized mean difference
For binary outcomes: risk ratio or odds ratio with 95% CI

Assess heterogeneity:
- I² < 25%: low
- I² 25-75%: moderate  
- I² > 75%: high

If heterogeneity is high, recommend subgroup analysis or narrative synthesis only.

Output JSON with pooled estimates and forest plot data."""


NARRATIVE_PROMPT = """You are a systematic review writer.

Transform the synthesized evidence into publication-ready narrative text.

Structure by PRISMA sections:
- Results: Findings organized by outcome/theme
- Discussion: Interpretation, limitations, implications

Writing guidelines:
- Use passive voice for objectivity
- Report exact numbers with CIs
- Cite studies inline (Author et al., Year)
- Acknowledge limitations honestly
- Avoid causal language unless RCT evidence

Output the narrative text with inline citations."""


# =============================================================================
# SYNTHESIS CLASSES
# =============================================================================


class EvidenceSynthesizer:
    """Synthesize evidence from multiple studies into coherent claims."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize synthesizer.

        Args:
            llm_provider: LLM provider for synthesis
            model: Model to use for synthesis
        """
        self.llm = llm_provider
        self.model = model

    def synthesize(
        self,
        study_cards: list[StudyCard],
        rob2_results: dict[str, RoB2Result] | list[RoB2Result],
        research_question: str,
        valid_snippet_ids: set[str] | None = None,
    ) -> SynthesisResult:
        """Synthesize evidence from multiple studies.

        Args:
            study_cards: List of extracted study cards
            rob2_results: RoB2 results (dict or list)
            research_question: Original research question
            valid_snippet_ids: Optional set of valid snippet IDs for early filtering.
                               If provided, claims with invalid snippet_ids are rejected
                               at parsing time, not at the gate.
                               Refs: PRISMA 2020 traceability, ADR-003

        Returns:
            SynthesisResult with claims and narrative
        """
        with tracer.start_span("synthesis.synthesize") as span:
            span.set_attribute("study_count", len(study_cards))
            if valid_snippet_ids is not None:
                span.set_attribute("valid_snippet_ids_provided", len(valid_snippet_ids))

            # Handle rob2_results as dict or list
            if isinstance(rob2_results, list):
                rob2_dict = {r.record_id: r for r in rob2_results}
            else:
                rob2_dict = rob2_results

            # Group studies by outcome type
            outcomes_by_type = self._group_by_outcome_type(study_cards)

            # Build context for LLM
            context = self._build_synthesis_context(study_cards, rob2_dict, research_question)

            # Generate synthesis
            messages = [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ]

            # Try structured output first for reliable JSON
            try:
                from cdr.llm.structured_outputs import get_synthesis_response_format

                response_format = get_synthesis_response_format()
            except ImportError:
                response_format = None

            # Use sync complete() method
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=4000,
                response_format=response_format,
            )

            # Parse response - pass valid_snippet_ids for early filtering
            result = self._parse_synthesis_response(
                response.content, study_cards, valid_snippet_ids
            )

            span.set_attribute("claim_count", len(result.claims))

            return result

    def _group_by_outcome_type(
        self,
        study_cards: list[StudyCard],
    ) -> dict[OutcomeMeasureType, list[tuple[StudyCard, OutcomeMeasure]]]:
        """Group outcomes by type for meta-analysis consideration."""
        grouped: dict[OutcomeMeasureType, list[tuple[StudyCard, OutcomeMeasure]]] = {}

        for card in study_cards:
            for outcome in card.outcomes:
                outcome_type = outcome.measure_type
                if outcome_type not in grouped:
                    grouped[outcome_type] = []
                grouped[outcome_type].append((card, outcome))

        return grouped

    def _build_synthesis_context(
        self,
        study_cards: list[StudyCard],
        rob2_results: dict[str, RoB2Result],
        research_question: str,
    ) -> str:
        """Build context string for LLM synthesis."""
        lines = [
            f"RESEARCH QUESTION: {research_question}",
            "",
            "INCLUDED STUDIES:",
            "",
        ]

        for i, card in enumerate(study_cards, 1):
            rob2 = rob2_results.get(card.record_id)

            lines.append(f"--- Study {i}: {card.record_id} ---")
            lines.append(f"Design: {card.study_type.value if card.study_type else 'Unknown'}")
            lines.append(f"Population: N={card.sample_size or 'NR'}")

            # Use extracted fields instead of pico
            if card.intervention_extracted:
                lines.append(f"Intervention: {card.intervention_extracted}")
            if card.comparator_extracted:
                lines.append(f"Comparator: {card.comparator_extracted}")

            if rob2:
                # CRITICAL: Distinguish genuine HIGH risk from error fallback
                # Per GRADE handbook: RoB2 errors should downgrade certainty
                # Refs: GRADE handbook section 5.2, Cochrane RoB2 tool
                if "ASSESSMENT FAILED" in (rob2.overall_rationale or ""):
                    lines.append(
                        f"Risk of Bias: {rob2.overall_judgment.value} (ASSESSMENT ERROR - CONSERVATIVE DEFAULT)"
                    )
                    lines.append(f"  ⚠️ RoB2 evaluation failed: certainty should be downgraded")
                else:
                    lines.append(f"Risk of Bias: {rob2.overall_judgment.value}")

            lines.append("Outcomes:")
            for outcome in card.outcomes:
                effect = ""
                if outcome.value is not None:
                    effect = f" (value={outcome.value}"
                    if outcome.ci_lower is not None and outcome.ci_upper is not None:
                        effect += f", 95% CI [{outcome.ci_lower}, {outcome.ci_upper}]"
                    if outcome.p_value is not None:
                        effect += f", p={outcome.p_value}"
                    effect += ")"
                lines.append(f"  - {outcome.name}: {outcome.measure_type.value}{effect}")

            lines.append("Key Findings:")
            # Use supporting_snippet_ids instead of snippets
            for snippet_id in card.supporting_snippet_ids[:3]:
                lines.append(f"  - Referenced: {snippet_id}")

            lines.append("")

        lines.append("Please synthesize the evidence and generate claims with GRADE assessments.")

        return "\n".join(lines)

    def _parse_synthesis_response(
        self,
        content: str,
        study_cards: list[StudyCard],
        valid_snippet_ids: set[str] | None = None,
    ) -> "SynthesisResult":
        """Parse LLM response into SynthesisResult.

        Maps LLM output to EvidenceClaim schema correctly:
        - claim_text (not statement)
        - supporting_snippet_ids (not supporting_study_ids/snippets)
        - limitations list

        Handles both JSON and Markdown outputs with robust fallback parsing.

        Args:
            content: Raw LLM response
            study_cards: Study cards for context
            valid_snippet_ids: If provided, filter snippet_ids to only those that exist.
                               This enables early rejection of claims with invalid snippets.
                               Refs: PRISMA 2020 traceability, ADR-003 post-audit
        """
        import re

        original_content = content
        content = content.strip()

        # === STAGE 1: Try JSON extraction ===

        # Try to extract JSON from markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        # Try to find JSON object in the content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)

        data = None
        json_parsed_successfully = False
        try:
            data = json.loads(content)
            json_parsed_successfully = True
            print(f"[Synthesize] Parsed JSON with {len(data.get('claims', []))} claims")
        except json.JSONDecodeError as e:
            print(f"[Synthesize] JSON parse failed: {e}")
            print(f"[Synthesize] Content (first 500 chars): {original_content[:500]}")

        # === STAGE 2: Markdown fallback parsing ===
        # CRITICAL: Only trigger fallback if JSON parsing FAILED
        # If JSON parsed but has no claims, that's a valid (empty) result, not fallback
        # Refs: ADR-005 DoD Level 2+ gate requires distinguishing JSON vs fallback

        if data is None or not data.get("claims"):
            # If JSON was parsed successfully but has no claims, return empty result (not fallback)
            if json_parsed_successfully and data is not None:
                return SynthesisResult(
                    claims=[],
                    heterogeneity_assessment=str(
                        data.get("heterogeneity_assessment", "No claims extracted")
                    ),
                    overall_narrative=data.get("overall_synthesis", ""),
                    used_markdown_fallback=False,  # JSON parsed - not a fallback
                )

            # Try to extract claims from Markdown format
            # CRITICAL: Pass valid_snippet_ids to unify validation across ALL paths
            # Refs: ADR-004 Audit v3 finding - fallback was creating "doomed" claims
            markdown_claims = self._parse_markdown_claims(
                original_content, study_cards, valid_snippet_ids
            )
            if markdown_claims:
                print(f"[Synthesize] Markdown fallback extracted {len(markdown_claims)} claims")
                return SynthesisResult(
                    claims=markdown_claims,
                    heterogeneity_assessment="Extracted from Markdown response",
                    overall_narrative=original_content,
                    used_markdown_fallback=True,  # CRITICAL: Flag for DoD Level 2+ gate
                )

            # Complete fallback - no claims extractable
            return SynthesisResult(
                claims=[],
                heterogeneity_assessment="Unable to parse synthesis response",
                overall_narrative=original_content,
                used_markdown_fallback=True,  # Also flagged - JSON parsing failed
            )

        # === STAGE 3: Build claims from parsed JSON ===

        # Build claims according to EvidenceClaim schema
        # CRITICAL: Per PRISMA 2020 / GRADE, claims MUST have traceable evidence
        # If valid_snippet_ids is provided, filter early to avoid creating claims
        # that will be rejected by the gate (reduces false negatives from timing issues)
        # Refs: ADR-003 post-audit, PRISMA 2020 traceability
        claims = []
        skipped_for_no_snippets = 0
        skipped_for_invalid_snippets = 0

        for claim_data in data.get("claims", []):
            # PRIORITY 1: Use supporting_snippet_ids directly if LLM provided them
            # This is the canonical field per EvidenceClaim schema
            supporting_snippet_ids = claim_data.get("supporting_snippet_ids", [])

            # PRIORITY 2: If not provided, try to extract snippet_ids from snippets array
            # CRITICAL: Accept ALL snippet_ids including _snip_0 (that's the REAL format)
            # Refs: graph.py parse_documents_node generates {record_id}_snip_0
            if not supporting_snippet_ids:
                for snip_data in claim_data.get("snippets", []):
                    snippet_id = snip_data.get("snippet_id", "")
                    if snippet_id:
                        supporting_snippet_ids.append(snippet_id)
                    else:
                        # Try to construct from source_ref if snippet_id not explicit
                        ref_data = snip_data.get("source_ref", {})
                        record_id = ref_data.get("record_id", "")
                        if record_id:
                            # Use real format: {record_id}_snip_0
                            supporting_snippet_ids.append(f"{record_id}_snip_0")

            # PRIORITY 3: Extract from supporting_studies using real snippet format
            # CRITICAL: Only generate if we don't have early validation OR if snippet exists
            if not supporting_snippet_ids:
                for study_id in claim_data.get("supporting_studies", []):
                    candidate_id = f"{study_id}_snip_0"
                    # If we have valid_snippet_ids, only add if it actually exists
                    if valid_snippet_ids is None or candidate_id in valid_snippet_ids:
                        supporting_snippet_ids.append(candidate_id)

            # EARLY VALIDATION: If valid_snippet_ids provided, filter to only existing ones
            # This prevents creating claims that will be rejected by the gate
            # Refs: ADR-003 post-audit finding about false negatives
            if valid_snippet_ids is not None and supporting_snippet_ids:
                valid_support = [sid for sid in supporting_snippet_ids if sid in valid_snippet_ids]
                if not valid_support and supporting_snippet_ids:
                    # Had candidates but none are valid - skip to avoid false negative
                    skipped_for_invalid_snippets += 1
                    continue
                supporting_snippet_ids = valid_support

            # CRITICAL: Skip claims without ANY support
            # Per GRADE handbook: claims require traceable evidence
            if not supporting_snippet_ids:
                skipped_for_no_snippets += 1
                continue  # Skip claims without support - DO NOT FABRICATE

            certainty_str = claim_data.get("certainty", "low")
            try:
                certainty = GRADECertainty(certainty_str)
            except ValueError:
                certainty = GRADECertainty.LOW

            # Extract limitations from grade_rationale if present
            limitations = []
            grade_rationale = claim_data.get("grade_rationale", {})
            if isinstance(grade_rationale, dict):
                for key, value in grade_rationale.items():
                    if value and "downgrade" in str(value).lower():
                        limitations.append(f"{key}: {value}")
            else:
                # Not a dict - reset to empty dict
                grade_rationale = {}

            # Get claim text (LLM may use 'statement' or 'claim_text')
            claim_text = claim_data.get("statement", "") or claim_data.get("claim_text", "")
            if len(claim_text) < 20:
                # Claim text too short, try to construct from context
                claim_text = (
                    f"Claim regarding {claim_data.get('claim_id', 'evidence')}: {claim_text}"
                )

            # Parse therapeutic_context from LLM response
            # CRITICAL: Must classify to prevent mixing contexts in synthesis
            # Refs: GRADE Handbook Section 5, Cochrane Handbook Section 11, CDR DoD P2
            context_str = claim_data.get("therapeutic_context", "unclassified")
            context_map = {
                "monotherapy": TherapeuticContext.MONOTHERAPY,
                "monotherapy_vs_active": TherapeuticContext.MONOTHERAPY_VS_ACTIVE,
                "add_on": TherapeuticContext.ADD_ON,
                "combination": TherapeuticContext.COMBINATION,
                "aspirin_monotherapy": TherapeuticContext.ASPIRIN_MONOTHERAPY,
                "aspirin_plus_anticoagulant": TherapeuticContext.ASPIRIN_PLUS_ANTICOAGULANT,
                "doac_vs_aspirin": TherapeuticContext.DOAC_VS_ASPIRIN,
                "doac_vs_warfarin": TherapeuticContext.DOAC_VS_WARFARIN,
                "head_to_head": TherapeuticContext.HEAD_TO_HEAD,
                "prevention": TherapeuticContext.PREVENTION,
                "treatment": TherapeuticContext.TREATMENT,
                "unclassified": TherapeuticContext.UNCLASSIFIED,
            }
            therapeutic_context = context_map.get(
                context_str.lower(), TherapeuticContext.UNCLASSIFIED
            )

            claims.append(
                EvidenceClaim(
                    claim_id=claim_data.get("claim_id", f"claim_{len(claims) + 1:03d}"),
                    claim_text=claim_text,
                    certainty=certainty,
                    therapeutic_context=therapeutic_context,
                    supporting_snippet_ids=supporting_snippet_ids,
                    conflicting_snippet_ids=[],  # To be populated by verification
                    limitations=limitations,
                    grade_rationale=grade_rationale,  # Structured GRADE rationale per ADR-004
                    studies_supporting=len(claim_data.get("supporting_studies", []))
                    or len(supporting_snippet_ids),
                    studies_conflicting=0,
                )
            )

        return SynthesisResult(
            claims=claims,
            heterogeneity_assessment=str(data.get("heterogeneity_assessment", {})),
            overall_narrative=data.get("overall_synthesis", ""),
            used_markdown_fallback=False,  # JSON path - no fallback used
        )

    def generate_narrative(
        self,
        synthesis_result: "SynthesisResult",
        section: Section = Section.RESULTS,
    ) -> str:
        """Generate publication-ready narrative from synthesis.

        Args:
            synthesis_result: Result from synthesize()
            section: Which section to generate

        Returns:
            Narrative text with inline citations
        """
        with tracer.start_span("synthesis.generate_narrative") as span:
            span.set_attribute("section", section.value)

            context = self._build_narrative_context(synthesis_result, section)

            messages = [
                {"role": "system", "content": NARRATIVE_PROMPT},
                {"role": "user", "content": context},
            ]

            # Use sync complete() method
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.4,
                max_tokens=3000,
            )

            return response.content

    def _build_narrative_context(
        self,
        synthesis_result: "SynthesisResult",
        section: Section,
    ) -> str:
        """Build context for narrative generation."""
        lines = [
            f"Generate the {section.value} section narrative.",
            "",
            "SYNTHESIZED CLAIMS:",
            "",
        ]

        for claim in synthesis_result.claims:
            # Use claim_text per EvidenceClaim schema
            lines.append(f"Claim: {claim.claim_text}")
            lines.append(f"Certainty: {claim.certainty.value}")
            # Use supporting_snippet_ids per EvidenceClaim schema
            lines.append(f"Supporting Evidence: {', '.join(claim.supporting_snippet_ids)}")
            lines.append(f"Studies Supporting: {claim.studies_supporting}")
            if claim.limitations:
                lines.append(f"Limitations: {'; '.join(claim.limitations)}")
            lines.append("")

        lines.append(f"Heterogeneity: {synthesis_result.heterogeneity_assessment}")
        lines.append("")
        lines.append(f"Overall: {synthesis_result.overall_narrative}")

        return "\n".join(lines)

    def _parse_markdown_claims(
        self,
        content: str,
        study_cards: list[StudyCard],
        valid_snippet_ids: set[str] | None = None,
    ) -> list[EvidenceClaim]:
        """Parse claims from Markdown-formatted LLM output.

        This is a fallback for when the LLM ignores response_format and
        returns Markdown instead of JSON.

        Args:
            content: Raw Markdown content from LLM
            study_cards: Study cards for context
            valid_snippet_ids: If provided, validate snippet_ids before creating claims.
                               Claims with NO valid snippets are skipped.
                               Refs: PRISMA 2020 traceability, ADR-004 Audit v3

        Handles patterns like:
        - **Claim 1:** or 1. **Claim ID:**
        - **Statement:** text
        - **Certainty:** moderate
        - **Supporting Studies:** [pmid:123]
        """
        import re

        claims = []

        # Split by claim markers (numbered claims or bold headers)
        claim_patterns = [
            r"\*\*Claim\s*\d+[:\.]?\*\*",  # **Claim 1:** or **Claim 1**
            r"\d+\.\s*\*\*Claim\s*ID[:\*]*",  # 1. **Claim ID:**
            r"#+\s*Claim\s*\d+",  # ## Claim 1
        ]

        combined_pattern = "|".join(f"({p})" for p in claim_patterns)
        splits = re.split(combined_pattern, content, flags=re.IGNORECASE)

        # Also try splitting by numbered list items if no claim headers found
        if len(splits) <= 1:
            splits = re.split(r"(?=\d+\.\s*\*\*)", content)

        for section in splits:
            if not section or len(section) < 50:
                continue

            # Extract claim components using various patterns
            claim_id = None
            statement = None
            certainty_str = "low"
            supporting_studies = []

            # Try to extract Claim ID
            id_match = re.search(r"claim_?\s*[iI][dD][:*\s]*([a-zA-Z0-9_]+)", section)
            if id_match:
                claim_id = id_match.group(1)

            # Try to extract Statement
            statement_patterns = [
                r"\*\*[Ss]tatement[:*\s]*\*?\*?\s*(.+?)(?=\n\*\*|\n\n|$)",
                r"[Ss]tatement[:]\s*(.+?)(?=\n[A-Z]|\n\*|\n\n|$)",
            ]
            for pattern in statement_patterns:
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    statement = match.group(1).strip()
                    # Clean up markdown
                    statement = re.sub(r"\*+", "", statement)
                    statement = statement.split("\n")[0].strip()
                    if len(statement) > 20:
                        break

            # If no statement found, try to use the whole section
            if not statement or len(statement) < 20:
                # Extract first substantial sentence
                sentences = re.split(r"[.!?]\s+", section)
                for sent in sentences:
                    clean = re.sub(r"\*+|#|^\d+\.?\s*", "", sent).strip()
                    if len(clean) > 30 and not clean.lower().startswith(
                        ("claim", "certainty", "supporting")
                    ):
                        statement = clean
                        break

            if not statement or len(statement) < 20:
                continue

            # Extract Certainty
            certainty_match = re.search(
                r"[Cc]ertainty[:*\s]*\*?\*?\s*(high|moderate|low|very[_\s]*low)",
                section,
                re.IGNORECASE,
            )
            if certainty_match:
                certainty_str = certainty_match.group(1).lower().replace(" ", "_")

            # Extract Supporting Studies
            studies_match = re.search(
                r"[Ss]upporting\s*[Ss]tudies[:*\s]*\[?([^\]]+)\]?", section, re.IGNORECASE
            )
            if studies_match:
                studies_text = studies_match.group(1)
                # Extract PMIDs or study IDs
                pmid_matches = re.findall(
                    r"(?:pmid[:\s]*)?([\d]{7,8})", studies_text, re.IGNORECASE
                )
                for pmid in pmid_matches:
                    supporting_studies.append(f"pubmed_{pmid}")

            # Create snippet IDs using REAL format (not placeholders)
            # CRITICAL: _snip_0 IS the real format generated by parse_documents_node
            # Gate in graph.py will validate existence in state.snippets
            # Refs: graph.py L493, PRISMA 2020 traceability
            if supporting_studies:
                # Generate candidate snippet_ids
                candidate_snippet_ids = [f"{s}_snip_0" for s in supporting_studies]

                # CRITICAL: Validate with valid_snippet_ids if provided
                # This prevents creating "doomed" claims that will be rejected by gate
                # Refs: ADR-004 Audit v3, PRISMA 2020 traceability
                if valid_snippet_ids is not None:
                    supporting_snippet_ids = [
                        sid for sid in candidate_snippet_ids if sid in valid_snippet_ids
                    ]
                    if not supporting_snippet_ids:
                        # All candidates are invalid - skip this claim
                        print(
                            f"[Synthesize] Markdown: Skipping claim - no valid snippets "
                            f"(candidates: {candidate_snippet_ids})"
                        )
                        continue
                else:
                    supporting_snippet_ids = candidate_snippet_ids
            else:
                # No supporting studies found - skip this claim
                # Per GRADE handbook: claims require traceable evidence
                continue  # Skip claims without any support

            # Parse certainty
            try:
                certainty = GRADECertainty(certainty_str.replace("_", "_"))
            except ValueError:
                certainty = GRADECertainty.LOW

            # Generate claim ID if not found
            if not claim_id:
                claim_id = f"claim_{len(claims) + 1:03d}"

            claims.append(
                EvidenceClaim(
                    claim_id=claim_id,
                    claim_text=statement,
                    certainty=certainty,
                    supporting_snippet_ids=supporting_snippet_ids,
                    conflicting_snippet_ids=[],
                    limitations=["Extracted from Markdown response - verify manually"],
                    # Markdown fallback has no structured rationale - to be assessed later
                    grade_rationale={},
                    studies_supporting=len(supporting_studies) or 1,
                    studies_conflicting=0,
                )
            )

        return claims


# =============================================================================
# QUANTITATIVE SYNTHESIS HELPERS
# =============================================================================


def calculate_pooled_estimate(
    effects: list[float],
    standard_errors: list[float],
    method: str = "fixed",
) -> dict:
    """Calculate pooled effect estimate.

    Args:
        effects: Effect sizes from studies
        standard_errors: Standard errors of effects
        method: 'fixed' or 'random' effects model

    Returns:
        Dict with pooled estimate, CI, and I²
    """
    if not effects or not standard_errors:
        return {"pooled": None, "ci_lower": None, "ci_upper": None, "i_squared": None}

    import math

    n = len(effects)

    # Calculate weights (inverse variance)
    weights = [1 / (se**2) if se > 0 else 0 for se in standard_errors]
    total_weight = sum(weights)

    if total_weight == 0:
        return {"pooled": None, "ci_lower": None, "ci_upper": None, "i_squared": None}

    # Fixed effects pooled estimate
    pooled = sum(w * e for w, e in zip(weights, effects)) / total_weight

    # Standard error of pooled estimate
    se_pooled = math.sqrt(1 / total_weight)

    # 95% CI
    z = 1.96
    ci_lower = pooled - z * se_pooled
    ci_upper = pooled + z * se_pooled

    # Calculate I² (heterogeneity)
    q = sum(w * (e - pooled) ** 2 for w, e in zip(weights, effects))
    df = n - 1
    i_squared = max(0, (q - df) / q * 100) if q > 0 else 0

    return {
        "pooled": round(pooled, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "se": round(se_pooled, 3),
        "i_squared": round(i_squared, 1),
        "n_studies": n,
        "method": method,
    }


def assess_publication_bias(
    effects: list[float],
    standard_errors: list[float],
) -> dict:
    """Assess publication bias using Egger's test approximation.

    Args:
        effects: Effect sizes
        standard_errors: Standard errors

    Returns:
        Dict with bias assessment
    """
    if len(effects) < 3:
        return {
            "assessment": "insufficient_studies",
            "detail": "Need at least 3 studies to assess publication bias",
        }

    # Simplified Egger's test (correlation between effect and precision)
    import math

    precisions = [1 / se if se > 0 else 0 for se in standard_errors]

    # Calculate correlation
    n = len(effects)
    mean_e = sum(effects) / n
    mean_p = sum(precisions) / n

    numerator = sum((e - mean_e) * (p - mean_p) for e, p in zip(effects, precisions))
    denom_e = math.sqrt(sum((e - mean_e) ** 2 for e in effects))
    denom_p = math.sqrt(sum((p - mean_p) ** 2 for p in precisions))

    if denom_e * denom_p == 0:
        correlation = 0
    else:
        correlation = numerator / (denom_e * denom_p)

    # Interpret
    if abs(correlation) < 0.3:
        assessment = "unlikely"
        detail = "No strong correlation between effect size and precision"
    elif abs(correlation) < 0.6:
        assessment = "possible"
        detail = "Moderate correlation suggests possible bias"
    else:
        assessment = "likely"
        detail = "Strong correlation between effect and precision suggests bias"

    return {
        "assessment": assessment,
        "detail": detail,
        "correlation": round(correlation, 3),
    }

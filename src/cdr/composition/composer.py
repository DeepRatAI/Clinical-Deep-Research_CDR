"""
Compositional Inference Engine

Generates novel hypotheses (A+B⇒C) from existing evidence.

HIGH-1 fix: Implementation of compositional inference.
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1

The engine:
1. Extracts mechanistic relations from claims
2. Builds a concept graph
3. Identifies composable claim pairs
4. Generates novel hypotheses with threat analysis
5. Proposes validation study designs

Scientific foundations:
- Causal representation learning: https://arxiv.org/abs/2102.11107
- Target trial emulation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4866534/
"""

import json
from typing import Any

from cdr.composition.schemas import (
    ComposedHypothesis,
    HypothesisStrength,
    MechanisticRelation,
    RelationType,
    ProposedStudyDesign,
    ThreatAnalysis,
)
from cdr.core.schemas import EvidenceClaim, PICO
from cdr.llm.base import BaseLLMProvider
from cdr.observability import get_tracer


tracer = get_tracer(__name__)


# Prompt for extracting mechanistic relations
RELATION_EXTRACTION_PROMPT = """You are a biomedical knowledge extractor. Analyze the following evidence claim and extract ALL mechanistic relations, including intermediate steps.

Claim: {claim_text}
PICO Context: {pico_context}

Extract structured relations in the following JSON format:
{{
    "relations": [
        {{
            "source_concept": "intervention, factor, or biological entity",
            "target_concept": "outcome, effect, or downstream entity",
            "mechanism": "biological pathway or mechanism if known, or null",
            "relation_type": "causal|associative|mechanistic|temporal|modulating|inhibitory",
            "confidence": 0.0 to 1.0 based on evidence strength
        }}
    ]
}}

IMPORTANT RULES:
- Extract ALL relations, including intermediate mechanistic steps
- If claim says "A causes B which leads to C", extract BOTH A→B AND B→C
- Use normalized concept names (e.g., "platelet aggregation" not "aggregation of platelets")
- Use "causal" only for RCT-level evidence
- Use "mechanistic" for biochemical pathway steps
- Use "associative" for observational evidence
- Include mechanism if explicitly stated or strongly implied
- Be conservative with confidence scores

Return ONLY valid JSON, no additional text.
"""

# Prompt for composing hypotheses
COMPOSITION_PROMPT = """You are a scientific hypothesis generator. Given two related evidence claims, compose a novel testable hypothesis.

Claim A: {claim_a}
Claim B: {claim_b}
Shared Concepts: {shared_concepts}
PICO Context: {pico_context}

Generate a composed hypothesis in the following JSON format:
{{
    "hypothesis_text": "If [mechanism/intervention], then [outcome], because [mechanistic chain] (A + B implies C)",
    "mechanistic_chain": [
        {{
            "source": "starting concept",
            "target": "intermediate or final outcome",
            "mechanism": "specific pathway connecting them (NOT 'unknown')"
        }}
    ],
    "strength": "strong|moderate|weak|exploratory",
    "confidence": 0.0 to 1.0,
    "rival_hypotheses": ["alternative explanation 1", "alternative explanation 2"],
    "uncontrolled_confounders": ["potential confounder 1"],
    "evidence_gaps": ["missing evidence 1"],
    "mcid_estimate": {{
        "value": numeric value or null,
        "unit": "measurement unit",
        "rationale": "basis for estimate (literature, clinical consensus)"
    }},
    "reasoning": "step-by-step reasoning for this composition"
}}

CRITICAL REQUIREMENTS (MANDATORY):
1. hypothesis_text MUST start with "If" and contain "then" (if-then structure)
2. mechanistic_chain MUST have at least one entry with non-empty mechanism
3. rival_hypotheses MUST have at least 2 alternatives
4. uncontrolled_confounders MUST have at least 1 confounder
5. evidence_gaps MUST identify at least 1 gap
6. mcid_estimate SHOULD have value and rationale if clinical outcome

If you cannot meet these requirements, still provide the JSON but with strength="exploratory" and confidence < 0.3.

Return ONLY valid JSON, no additional text."""

# Few-shot example for better test design generation
TEST_DESIGN_FEW_SHOT = """EXAMPLE:
Hypothesis: "If GLP-1 agonists are added to metformin in T2DM patients, then CV events will decrease by 15%, because GLP-1 improves endothelial function independently of glycemic control"

Test Design:
{{
    "proposed_population": "Adults 40-75 years with T2DM, HbA1c 7-10%, on stable metformin ≥3 months, no prior CV event",
    "proposed_intervention": "Liraglutide 1.8mg daily or semaglutide 1mg weekly added to metformin",
    "proposed_comparator": "Placebo injection + metformin continuation",
    "proposed_outcome": "Time to first MACE (CV death, non-fatal MI, non-fatal stroke)",
    "mcid_value": 0.85,
    "mcid_rationale": "HR of 0.85 based on LEADER trial precedent; represents 15% relative risk reduction",
    "recommended_design": "RCT",
    "minimum_sample_size": 9340,
    "follow_up_duration": "3.8 years median",
    "critical_measurements": ["HbA1c quarterly", "CV biomarkers", "MACE adjudication", "eGFR", "adverse events"],
    "blinding_requirements": "double-blind"
}}

END EXAMPLE

"""

# Prompt for generating test design
TEST_DESIGN_PROMPT = """You are a clinical trial methodologist. Propose a study design to test this hypothesis.

{few_shot}

Hypothesis: {hypothesis}
Available Evidence Grade: {evidence_grade}
Original PICO: {pico_context}

Generate a test design in the following JSON format:
{{
    "proposed_population": "target population with inclusion/exclusion criteria",
    "proposed_intervention": "specific intervention details with dosing",
    "proposed_comparator": "control/comparator details",
    "proposed_outcome": "primary outcome with measurement method",
    "mcid_value": numeric value (REQUIRED - estimate based on literature or clinical consensus),
    "mcid_rationale": "justification for MCID (REQUIRED)",
    "recommended_design": "RCT|pragmatic_trial|observational|other",
    "minimum_sample_size": estimated N based on MCID and power 80%,
    "follow_up_duration": "recommended follow-up period",
    "critical_measurements": ["essential measurement 1", "essential measurement 2"],
    "blinding_requirements": "double-blind|single-blind|open-label|not_applicable"
}}

CRITICAL REQUIREMENTS:
1. mcid_value MUST be provided (numeric estimate based on clinical relevance)
2. mcid_rationale MUST explain the basis for this MCID
3. minimum_sample_size MUST be calculated based on MCID
4. All fields are REQUIRED - do not leave as null

Return ONLY valid JSON, no additional text.
"""


class CompositionEngine:
    """
    Engine for compositional inference (A+B⇒C).

    Takes evidence claims and generates novel, testable hypotheses
    by identifying composable claim pairs and their mechanistic links.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        model: str = "gpt-4o",
    ):
        """Initialize composition engine.

        Args:
            provider: LLM provider for inference (optional)
            model: Model name if creating provider internally
        """
        self.provider = provider
        self.model = model
        self._tracer = tracer

    def _get_provider(self) -> BaseLLMProvider | None:
        """Get or create LLM provider.

        Returns:
            LLM provider or None if creation fails
        """
        if self.provider is not None:
            return self.provider

        # Create provider on demand
        try:
            from cdr.llm import create_provider

            return create_provider(self.model)
        except Exception:
            # Provider creation failed (unknown model, missing credentials, etc.)
            return None

    def extract_relations(
        self,
        claims: list[EvidenceClaim],
        pico: PICO | None = None,
    ) -> list[MechanisticRelation]:
        """Extract mechanistic relations from claims.

        Args:
            claims: Evidence claims to analyze
            pico: Optional PICO context

        Returns:
            List of extracted mechanistic relations
        """
        with self._tracer.span("composition.extract_relations") as span:
            span.set_attribute("claims_count", len(claims))

            pico_context = ""
            if pico:
                pico_context = f"Population: {pico.population}, Intervention: {pico.intervention}, Outcome: {pico.outcome}"

            relations = []
            provider = self._get_provider()
            if provider is None:
                span.set_attribute("error", "no_provider")
                return []

            for claim in claims:
                prompt = RELATION_EXTRACTION_PROMPT.format(
                    claim_text=claim.claim_text,
                    pico_context=pico_context or "Not specified",
                )

                try:
                    llm_response = provider.complete(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=1500,  # Increased for detailed mechanistic chains
                    )
                    response = llm_response.content

                    # Parse JSON response
                    data = self._parse_json_response(response)
                    if data and "relations" in data:
                        for rel_data in data["relations"]:
                            relation = MechanisticRelation(
                                relation_id=f"rel_{claim.claim_id}_{len(relations)}",
                                source_concept=rel_data.get("source_concept", ""),
                                target_concept=rel_data.get("target_concept", ""),
                                mechanism=rel_data.get("mechanism"),
                                relation_type=RelationType(
                                    rel_data.get("relation_type", "associative")
                                ),
                                supporting_claim_ids=[claim.claim_id],
                                supporting_snippet_ids=claim.supporting_snippet_ids,
                                confidence_score=float(rel_data.get("confidence", 0.5)),
                            )
                            relations.append(relation)
                except Exception as e:
                    span.set_attribute(f"error_{claim.claim_id}", str(e))
                    continue

            span.set_attribute("relations_extracted", len(relations))
            return relations

    def find_composable_pairs(
        self,
        claims: list[EvidenceClaim],
        relations: list[MechanisticRelation],
    ) -> list[tuple[EvidenceClaim, EvidenceClaim, list[str]]]:
        """Find claim pairs that can be composed.

        Claims are composable if they share concepts that form a chain:
        Claim A: X → Y
        Claim B: Y → Z
        Composed: X → Y → Z (X may lead to Z via Y)

        Args:
            claims: All evidence claims
            relations: Extracted relations

        Returns:
            List of (claim_a, claim_b, shared_concepts) tuples
        """
        with self._tracer.span("composition.find_pairs") as span:
            # Build concept index from relations with normalization
            claim_to_targets: dict[str, set[str]] = {}
            claim_to_sources: dict[str, set[str]] = {}

            def normalize_concept(concept: str) -> str:
                """Normalize concept for matching."""
                c = concept.lower().strip()
                # Remove common prefixes/suffixes that don't change meaning
                for prefix in [
                    "high ",
                    "low ",
                    "increased ",
                    "decreased ",
                    "elevated ",
                    "reduced ",
                ]:
                    if c.startswith(prefix):
                        c = c[len(prefix) :]
                # Standardize common terms
                c = c.replace(" levels", "").replace(" level", "")
                c = c.replace(" reduction", "").replace(" increase", "")
                return c.strip()

            for rel in relations:
                for claim_id in rel.supporting_claim_ids:
                    if claim_id not in claim_to_targets:
                        claim_to_targets[claim_id] = set()
                        claim_to_sources[claim_id] = set()
                    claim_to_targets[claim_id].add(normalize_concept(rel.target_concept))
                    claim_to_sources[claim_id].add(normalize_concept(rel.source_concept))

            # Find pairs where one claim's target matches another's source
            pairs = []
            claim_map = {c.claim_id: c for c in claims}

            for claim_a_id, targets_a in claim_to_targets.items():
                for claim_b_id, sources_b in claim_to_sources.items():
                    if claim_a_id == claim_b_id:
                        continue

                    # Find shared concepts (A's targets that are B's sources)
                    # Use exact match after normalization
                    shared = targets_a & sources_b

                    # Also check for partial matches (one contains the other)
                    if not shared:
                        for t in targets_a:
                            for s in sources_b:
                                if t in s or s in t or self._concepts_similar(t, s):
                                    shared = {f"{t} ~ {s}"}
                                    break
                            if shared:
                                break

                    if shared:
                        claim_a = claim_map.get(claim_a_id)
                        claim_b = claim_map.get(claim_b_id)
                        if claim_a and claim_b:
                            pairs.append((claim_a, claim_b, list(shared)))

            span.set_attribute("composable_pairs", len(pairs))
            return pairs

    def _concepts_similar(self, concept_a: str, concept_b: str) -> bool:
        """Check if two concepts are semantically similar.

        Uses simple heuristics for common biomedical terms.
        """
        # Common term mappings
        synonyms = {
            "hba1c": {"hemoglobin a1c", "glycated hemoglobin", "a1c", "glycosylated hemoglobin"},
            "cardiovascular": {"cv", "heart", "cardiac"},
            "glucose": {"blood sugar", "glycemia"},
            "insulin": {"insulin secretion", "beta cell"},
            "atherosclerosis": {"arterial plaque", "arteriosclerosis"},
        }

        a_lower = concept_a.lower()
        b_lower = concept_b.lower()

        # Check synonym mappings
        for key, values in synonyms.items():
            if key in a_lower or any(v in a_lower for v in values):
                if key in b_lower or any(v in b_lower for v in values):
                    return True

        # Check word overlap (at least 50% words match)
        words_a = set(a_lower.split())
        words_b = set(b_lower.split())
        if words_a and words_b:
            overlap = len(words_a & words_b)
            if overlap / min(len(words_a), len(words_b)) >= 0.5:
                return True

        return False

    def compose_hypothesis(
        self,
        claim_a: EvidenceClaim,
        claim_b: EvidenceClaim,
        shared_concepts: list[str],
        pico: PICO | None = None,
        max_retries: int = 2,
    ) -> ComposedHypothesis | None:
        """Compose a novel hypothesis from two claims.

        Args:
            claim_a: First claim (provides premise)
            claim_b: Second claim (provides consequence)
            shared_concepts: Concepts linking the claims
            pico: Optional PICO context
            max_retries: Number of retries on validation failure

        Returns:
            Composed hypothesis or None if composition fails
        """
        with self._tracer.span("composition.compose") as span:
            span.set_attribute("claim_a", claim_a.claim_id)
            span.set_attribute("claim_b", claim_b.claim_id)

            pico_context = ""
            if pico:
                pico_context = f"Population: {pico.population}, Intervention: {pico.intervention}, Outcome: {pico.outcome}"

            prompt = COMPOSITION_PROMPT.format(
                claim_a=claim_a.claim_text,
                claim_b=claim_b.claim_text,
                shared_concepts=", ".join(shared_concepts),
                pico_context=pico_context or "Not specified",
            )

            provider = self._get_provider()
            if provider is None:
                span.set_attribute("error", "no_provider")
                return None

            for attempt in range(max_retries + 1):
                try:
                    # Use lower temperature on retries for more deterministic output
                    temperature = 0.2 if attempt > 0 else 0.3

                    llm_response = provider.complete(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1200,
                    )
                    response = llm_response.content

                    data = self._parse_json_response(response)
                    if not data:
                        span.set_attribute(f"attempt_{attempt}_error", "json_parse_failed")
                        continue

                    # HARDENING: Validate critical fields
                    validation_issues = self._validate_hypothesis_data(data)
                    if validation_issues:
                        span.set_attribute(f"attempt_{attempt}_validation", str(validation_issues))
                        if attempt < max_retries:
                            continue  # Retry with lower temperature
                        # On final attempt, proceed but mark as low quality
                        data["strength"] = "exploratory"
                        data["confidence"] = min(data.get("confidence", 0.3), 0.3)

                    # Build mechanistic chain
                    chain = []
                    for i, link in enumerate(data.get("mechanistic_chain", [])):
                        chain.append(
                            MechanisticRelation(
                                relation_id=f"chain_{claim_a.claim_id}_{claim_b.claim_id}_{i}",
                                source_concept=link.get("source", ""),
                                target_concept=link.get("target", ""),
                                mechanism=link.get("mechanism"),
                                relation_type=RelationType.MECHANISTIC,
                                supporting_claim_ids=[claim_a.claim_id, claim_b.claim_id],
                            )
                        )

                    # Build threat analysis
                    threat_analysis = ThreatAnalysis(
                        rival_hypotheses=data.get("rival_hypotheses", []),
                        uncontrolled_confounders=data.get("uncontrolled_confounders", []),
                        evidence_gaps=data.get("evidence_gaps", []),
                    )

                    # Map strength string to enum
                    strength_map = {
                        "strong": HypothesisStrength.STRONG,
                        "moderate": HypothesisStrength.MODERATE,
                        "weak": HypothesisStrength.WEAK,
                        "exploratory": HypothesisStrength.EXPLORATORY,
                    }
                    strength = strength_map.get(
                        data.get("strength", "weak"), HypothesisStrength.WEAK
                    )

                    hypothesis = ComposedHypothesis(
                        hypothesis_id=f"hyp_{claim_a.claim_id}_{claim_b.claim_id}",
                        hypothesis_text=data.get("hypothesis_text", ""),
                        source_claim_ids=[claim_a.claim_id, claim_b.claim_id],
                        mechanistic_chain=chain,
                        strength=strength,
                        confidence_score=float(data.get("confidence", 0.5)),
                        threat_analysis=threat_analysis,
                        reasoning_trace=data.get("reasoning"),
                    )

                    span.set_attribute("hypothesis_strength", strength.value)
                    span.set_attribute(
                        "validation_issues", str(validation_issues) if validation_issues else "none"
                    )
                    return hypothesis

                except Exception as e:
                    span.set_attribute(f"attempt_{attempt}_error", str(e))
                    if attempt >= max_retries:
                        return None

            return None

    def _validate_hypothesis_data(self, data: dict) -> list[str]:
        """Validate hypothesis JSON has all critical fields.

        Returns list of validation issues (empty if valid).
        """
        issues = []

        # Check if-then structure
        hypothesis_text = data.get("hypothesis_text", "")
        if not (hypothesis_text.lower().startswith("if") and "then" in hypothesis_text.lower()):
            issues.append("missing_if_then_structure")

        # Check mechanistic chain
        chain = data.get("mechanistic_chain", [])
        if not chain:
            issues.append("missing_mechanistic_chain")
        else:
            has_mechanism = any(link.get("mechanism") for link in chain)
            if not has_mechanism:
                issues.append("no_mechanisms_specified")

        # Check rival hypotheses (need at least 2)
        rivals = data.get("rival_hypotheses", [])
        if len(rivals) < 2:
            issues.append(f"insufficient_rivals_{len(rivals)}_need_2")

        # Check confounders (need at least 1)
        confounders = data.get("uncontrolled_confounders", [])
        if len(confounders) < 1:
            issues.append("missing_confounders")

        # Check evidence gaps (need at least 1)
        gaps = data.get("evidence_gaps", [])
        if len(gaps) < 1:
            issues.append("missing_evidence_gaps")

        return issues

    def propose_test_design(
        self,
        hypothesis: ComposedHypothesis,
        pico: PICO | None = None,
        max_retries: int = 2,
    ) -> ProposedStudyDesign | None:
        """Propose a study design to validate the hypothesis.

        Args:
            hypothesis: The composed hypothesis
            pico: Original PICO for context
            max_retries: Number of retries on validation failure

        Returns:
            Proposed test design or None if generation fails
        """
        with self._tracer.span("composition.test_design") as span:
            span.set_attribute("hypothesis_id", hypothesis.hypothesis_id)

            pico_context = ""
            if pico:
                pico_context = f"Population: {pico.population}, Intervention: {pico.intervention}, Outcome: {pico.outcome}"

            prompt = TEST_DESIGN_PROMPT.format(
                few_shot=TEST_DESIGN_FEW_SHOT,
                hypothesis=hypothesis.hypothesis_text,
                evidence_grade=hypothesis.strength.value,
                pico_context=pico_context or "Not specified",
            )

            provider = self._get_provider()
            if provider is None:
                span.set_attribute("error", "no_provider")
                return None

            for attempt in range(max_retries + 1):
                try:
                    # Use lower temperature on retries
                    temperature = 0.1 if attempt > 0 else 0.2

                    llm_response = provider.complete(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1000,
                    )
                    response = llm_response.content

                    data = self._parse_json_response(response)
                    if not data:
                        span.set_attribute(f"attempt_{attempt}_error", "json_parse_failed")
                        continue

                    # HARDENING: Validate critical test design fields
                    validation_issues = self._validate_test_design_data(data)
                    if validation_issues:
                        span.set_attribute(f"attempt_{attempt}_validation", str(validation_issues))
                        if attempt < max_retries:
                            continue
                        # On final attempt, infer missing values
                        data = self._infer_missing_test_design_fields(data)

                    test_design = ProposedStudyDesign(
                        proposed_population=data.get("proposed_population", ""),
                        proposed_intervention=data.get("proposed_intervention", ""),
                        proposed_comparator=data.get("proposed_comparator", ""),
                        proposed_outcome=data.get("proposed_outcome", ""),
                        mcid_value=data.get("mcid_value"),
                        mcid_rationale=data.get("mcid_rationale"),
                        recommended_design=data.get("recommended_design", "RCT"),
                        minimum_sample_size=data.get("minimum_sample_size"),
                        follow_up_duration=data.get("follow_up_duration"),
                        critical_measurements=data.get("critical_measurements", []),
                        blinding_requirements=data.get("blinding_requirements"),
                    )

                    span.set_attribute("design", test_design.recommended_design)
                    span.set_attribute("has_mcid", test_design.mcid_value is not None)
                    return test_design

                except Exception as e:
                    span.set_attribute(f"attempt_{attempt}_error", str(e))
                    if attempt >= max_retries:
                        return None

            return None

    def _validate_test_design_data(self, data: dict) -> list[str]:
        """Validate test design JSON has critical fields.

        Returns list of validation issues (empty if valid).
        """
        issues = []

        if not data.get("mcid_value"):
            issues.append("missing_mcid_value")

        if not data.get("mcid_rationale"):
            issues.append("missing_mcid_rationale")

        if not data.get("minimum_sample_size"):
            issues.append("missing_sample_size")

        if not data.get("proposed_population"):
            issues.append("missing_population")

        if not data.get("proposed_intervention"):
            issues.append("missing_intervention")

        return issues

    def _infer_missing_test_design_fields(self, data: dict) -> dict:
        """Infer missing test design fields with heuristics.

        Marks inferred fields in mcid_rationale.
        """
        data = dict(data)  # Copy to avoid mutation

        if not data.get("mcid_value"):
            # Default MCID based on outcome type
            data["mcid_value"] = 0.80  # Conservative HR for CV outcomes
            rationale = data.get("mcid_rationale", "")
            data["mcid_rationale"] = f"[INFERRED] Default HR=0.80 (20% RRR). {rationale}"

        if not data.get("minimum_sample_size"):
            # Estimate based on MCID: n = 2 * (Z_α + Z_β)² / (log(HR))²
            # For HR=0.80, power=80%, alpha=0.05: ~750 per arm
            data["minimum_sample_size"] = 1500

        return data

    def run(
        self,
        claims: list[EvidenceClaim],
        pico: PICO | None = None,
        max_hypotheses: int = 3,
        include_test_designs: bool = True,
    ) -> list[ComposedHypothesis]:
        """Run full compositional inference pipeline.

        Args:
            claims: Evidence claims to compose
            pico: Original PICO context
            max_hypotheses: Maximum number of hypotheses to generate
            include_test_designs: Whether to generate test designs

        Returns:
            List of composed hypotheses with threat analysis and test designs
        """
        with self._tracer.span("composition.run") as span:
            span.set_attribute("claims_count", len(claims))
            span.set_attribute("max_hypotheses", max_hypotheses)

            if len(claims) < 2:
                span.set_attribute("status", "insufficient_claims")
                return []

            # Step 1: Extract relations
            relations = self.extract_relations(claims, pico)
            if not relations:
                span.set_attribute("status", "no_relations")
                return []

            # Step 2: Find composable pairs
            pairs = self.find_composable_pairs(claims, relations)
            if not pairs:
                span.set_attribute("status", "no_composable_pairs")
                return []

            # Step 3: Compose hypotheses
            hypotheses = []
            for claim_a, claim_b, shared in pairs[: max_hypotheses * 2]:  # Try more pairs
                hyp = self.compose_hypothesis(claim_a, claim_b, shared, pico)
                if hyp:
                    # Step 4: Add test design if requested
                    if include_test_designs:
                        test = self.propose_test_design(hyp, pico)
                        if test:
                            # Create new hypothesis with test design
                            hyp = ComposedHypothesis(
                                hypothesis_id=hyp.hypothesis_id,
                                hypothesis_text=hyp.hypothesis_text,
                                source_claim_ids=hyp.source_claim_ids,
                                mechanistic_chain=hyp.mechanistic_chain,
                                strength=hyp.strength,
                                confidence_score=hyp.confidence_score,
                                threat_analysis=hyp.threat_analysis,
                                proposed_test=test,
                                reasoning_trace=hyp.reasoning_trace,
                            )

                    hypotheses.append(hyp)

                    if len(hypotheses) >= max_hypotheses:
                        break

            span.set_attribute("hypotheses_generated", len(hypotheses))
            return hypotheses

    def _parse_json_response(self, response: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, handling code blocks."""
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try finding JSON object
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        return None

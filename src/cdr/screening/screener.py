"""
Screener

Automatic screening of records based on PICO criteria.
Uses LLM for intelligent inclusion/exclusion decisions.
"""

import json
from typing import Any, TYPE_CHECKING

from cdr.config import get_settings
from cdr.core.enums import ExclusionReason, StudyType
from cdr.core.schemas import PICO, Record, ScreeningDecision
from cdr.llm import Message, create_provider, build_messages
from cdr.observability import get_tracer, get_cdr_metrics

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


SCREENING_SYSTEM_PROMPT = """You are a systematic review screening assistant. Your task is to evaluate whether a study should be INCLUDED or EXCLUDED based on the PICO criteria provided.

CRITICAL RULES:
1. You MUST provide a decision: INCLUDE or EXCLUDE
2. If EXCLUDED, you MUST provide an exclusion_reason from the valid list
3. You MUST provide a rationale explaining your decision
4. Be conservative: when in doubt, INCLUDE for human review
5. Base your decision ONLY on the information provided

Valid exclusion reasons (use EXACTLY these values):
- population_mismatch: Study population doesn't match criteria
- intervention_mismatch: Intervention doesn't match criteria
- outcome_mismatch: Outcomes don't match criteria
- study_type_excluded: Not the required study design (e.g., not an RCT)
- duplicate: Duplicate publication
- language_not_supported: Not in English
- no_abstract: No abstract available
- animal_study: Animal study, not human
- in_vitro_only: In vitro study only
- retracted: Retracted publication
- other: Other reason (must explain)

Output your response as valid JSON:
{
    "decision": "INCLUDE" or "EXCLUDE",
    "exclusion_reason": null or one of the valid reasons,
    "rationale": "Your reasoning...",
    "confidence": 0.0 to 1.0
}"""


def _build_screening_prompt(pico: PICO, record: Record) -> str:
    """Build the screening prompt for a record.

    CRITICAL: Uses only fields that exist in Record schema (source of truth).
    """
    # Record doesn't have study_type field - use publication_type if available
    pub_types = ", ".join(record.publication_type) if record.publication_type else "Unknown"

    return f"""## PICO Criteria

**Population:** {pico.population}
**Intervention:** {pico.intervention}
**Comparator:** {pico.comparator}
**Outcome:** {pico.outcome}
**Required Study Types:** {", ".join(st.value for st in pico.study_types)}

## Study to Screen

**ID:** {record.record_id}
**Title:** {record.title}
**Abstract:** {record.abstract or "No abstract available"}
**Publication Types:** {pub_types}
**Year:** {record.year or "Unknown"}
**Journal:** {record.journal or "Unknown"}
**Source:** {record.source.value}

Evaluate this study against the PICO criteria and provide your decision."""


class Screener:
    """
    Automatic study screening based on PICO criteria.

    Usage:
        screener = Screener()
        decisions = screener.screen_records(pico, records)
    """

    def __init__(
        self,
        provider: "BaseLLMProvider | str | None" = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize screener.

        Args:
            provider: LLM provider instance or name. If instance, use directly.
                      If string/None, creates one using create_provider().
            model: Model name (used only if provider is None or string).
            temperature: Sampling temperature (0 = deterministic).

        Refs: CDR_Integral_Audit_2026-01-20.md CRITICAL-2
        """
        from cdr.llm.base import BaseLLMProvider

        # Accept both provider instances and provider names
        if provider is not None and isinstance(provider, BaseLLMProvider):
            self._provider = provider
        else:
            # Fallback: create provider if not passed or if string name
            provider_name = provider if isinstance(provider, str) else None
            self._provider = create_provider(provider=provider_name, model=model)
        self._temperature = temperature
        self._tracer = get_tracer("cdr.screening")
        self._metrics = get_cdr_metrics()

    def screen_record(self, pico: PICO, record: Record) -> ScreeningDecision:
        """
        Screen a single record.

        Args:
            pico: PICO criteria.
            record: Record to screen.

        Returns:
            Screening decision.
        """
        with self._tracer.span("screen_record", attributes={"record_id": record.record_id}) as span:
            messages = build_messages(
                system=SCREENING_SYSTEM_PROMPT,
                user=_build_screening_prompt(pico, record),
            )

            try:
                response = self._provider.complete(
                    messages,
                    temperature=self._temperature,
                    max_tokens=500,
                )

                self._metrics.llm_requests.inc(labels={"operation": "screening"})
                self._metrics.llm_tokens.inc(
                    response.total_tokens, labels={"operation": "screening"}
                )

                decision = self._parse_response(record.record_id, response.content)
                span.set_attribute("decision", "included" if decision.included else "excluded")

                return decision

            except Exception as e:
                span.set_status("error", str(e))
                # On error, default to include for human review
                return ScreeningDecision(
                    record_id=record.record_id,
                    included=True,
                    exclusion_rationale=f"Screening error, included for review: {e}",
                    confidence=0.0,
                )

    def screen_records(
        self,
        pico: PICO,
        records: list[Record],
        batch_size: int = 10,
    ) -> list[ScreeningDecision]:
        """
        Screen multiple records.

        Args:
            pico: PICO criteria.
            records: Records to screen.
            batch_size: Batch size for processing.

        Returns:
            List of screening decisions.
        """
        with self._tracer.span("screen_records", attributes={"count": len(records)}) as span:
            decisions: list[ScreeningDecision] = []

            for i, record in enumerate(records):
                decision = self.screen_record(pico, record)
                decisions.append(decision)

                self._metrics.records_screened.inc(
                    labels={"decision": "included" if decision.included else "excluded"}
                )

            included = sum(1 for d in decisions if d.included)
            excluded = len(decisions) - included

            span.set_attribute("included", included)
            span.set_attribute("excluded", excluded)

            return decisions

    def _parse_response(self, record_id: str, content: str) -> ScreeningDecision:
        """Parse LLM response into ScreeningDecision."""
        try:
            # Clean up response
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)

            decision = data.get("decision", "").upper()
            included = decision == "INCLUDE"

            exclusion_reason = None
            if not included:
                reason_str = data.get("exclusion_reason")
                if reason_str:
                    try:
                        exclusion_reason = ExclusionReason(reason_str)
                    except ValueError:
                        exclusion_reason = ExclusionReason.OTHER

            return ScreeningDecision(
                record_id=record_id,
                included=included,
                exclusion_reason=exclusion_reason,
                exclusion_rationale=data.get("rationale"),
                confidence=data.get("confidence"),
            )

        except json.JSONDecodeError:
            # Parse failure - include for review
            return ScreeningDecision(
                record_id=record_id,
                included=True,
                exclusion_rationale="Failed to parse screening response, included for review",
                confidence=0.0,
            )


class RuleBasedScreener:
    """
    Fast rule-based pre-screening for obvious exclusions.

    Use before LLM screening to reduce costs.
    """

    def __init__(self, pico: PICO) -> None:
        """Initialize with PICO criteria."""
        self._pico = pico
        self._tracer = get_tracer("cdr.screening.rules")

    def prescreen(self, records: list[Record]) -> tuple[list[Record], list[ScreeningDecision]]:
        """
        Apply rule-based pre-screening.

        Returns:
            Tuple of (records_for_llm_screening, rule_based_exclusions).
        """
        with self._tracer.span("prescreen", attributes={"count": len(records)}) as span:
            to_screen: list[Record] = []
            excluded: list[ScreeningDecision] = []

            for record in records:
                decision = self._apply_rules(record)
                if decision:
                    excluded.append(decision)
                else:
                    to_screen.append(record)

            span.set_attribute("excluded_by_rules", len(excluded))
            span.set_attribute("for_llm_screening", len(to_screen))

            return to_screen, excluded

    def _apply_rules(self, record: Record) -> ScreeningDecision | None:
        """Apply exclusion rules. Returns decision if excluded, None if passes.

        CRITICAL: Record schema does NOT have 'study_type' or 'id' fields.
        - Use record.record_id (not record.id)
        - Infer study type from publication_type (PubMed) when needed

        Refs: NCBI E-utilities, ClinicalTrials.gov API
        """
        # Infer study type from publication_type if available
        # PubMed uses publication_type list (e.g., ['Randomized Controlled Trial'])
        inferred_study_type = None
        if record.publication_type:
            pub_types_lower = [pt.lower() for pt in record.publication_type]
            if any("randomized controlled trial" in pt for pt in pub_types_lower):
                inferred_study_type = StudyType.RCT
            elif any("systematic review" in pt for pt in pub_types_lower):
                inferred_study_type = StudyType.SYSTEMATIC_REVIEW
            elif any("meta-analysis" in pt for pt in pub_types_lower):
                inferred_study_type = StudyType.META_ANALYSIS
            elif any("cohort" in pt for pt in pub_types_lower):
                inferred_study_type = StudyType.COHORT
            elif any("case-control" in pt for pt in pub_types_lower):
                inferred_study_type = StudyType.CASE_CONTROL

        # Check study type if we could infer it
        if inferred_study_type and self._pico.study_types:
            if inferred_study_type not in self._pico.study_types:
                return ScreeningDecision(
                    record_id=record.record_id,
                    included=False,
                    exclusion_reason=ExclusionReason.STUDY_TYPE_EXCLUDED,
                    exclusion_rationale=f"Study type {inferred_study_type.value} not in required types",
                    confidence=1.0,
                )

        # Check for animal study keywords
        if record.abstract:
            abstract_lower = record.abstract.lower()
            animal_keywords = [
                "mice",
                "rats",
                "mouse",
                "rat",
                "murine",
                "porcine",
                "canine",
                "animal model",
            ]
            if any(kw in abstract_lower for kw in animal_keywords):
                # Check if it's not also a human study
                human_keywords = ["patients", "participants", "subjects", "human", "clinical trial"]
                if not any(kw in abstract_lower for kw in human_keywords):
                    return ScreeningDecision(
                        record_id=record.record_id,
                        included=False,
                        exclusion_reason=ExclusionReason.ANIMAL_STUDY,
                        exclusion_rationale="Appears to be animal study based on keywords",
                        confidence=0.8,
                    )

        return None

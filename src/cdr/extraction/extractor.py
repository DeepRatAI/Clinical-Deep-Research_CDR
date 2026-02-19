"""
StudyCard Extractor

Structured extraction of study metadata using DSPy.
"""

import json
from typing import Any

from cdr.config import get_settings
from cdr.core.enums import OutcomeMeasureType, StudyType
from cdr.core.schemas import OutcomeMeasure, Snippet, SourceRef, StudyCard
from cdr.core.exceptions import ExtractionError
from cdr.llm import Message, create_provider, build_messages
from cdr.observability import get_tracer, get_cdr_metrics


EXTRACTION_SYSTEM_PROMPT = """You are a clinical research data extraction specialist. Extract structured study information from the provided text.

CRITICAL RULES:
1. Extract ONLY information explicitly stated in the text
2. Mark fields as null if information is not available
3. For statistical outcomes, include exact values with confidence intervals when available
4. Record the exact text snippets that support each extracted field
5. Be precise with population counts - don't estimate

For outcome measures, use these types:
- MEAN_DIFFERENCE: Difference between means
- RISK_RATIO: Relative risk (RR)
- ODDS_RATIO: Odds ratio (OR)
- HAZARD_RATIO: Hazard ratio (HR)
- RISK_DIFFERENCE: Absolute risk difference
- PERCENTAGE: Percentage value
- COUNT: Raw count
- OTHER: Other measure type

Output your response as valid JSON matching this schema:
{
    "study_type": "RCT" | "META_ANALYSIS" | "SYSTEMATIC_REVIEW" | "COHORT" | "CASE_CONTROL" | "CROSS_SECTIONAL" | "CASE_SERIES" | "CASE_REPORT" | null,
    "population_n": number or null,
    "population_description": "string or null",
    "intervention_description": "string or null",
    "comparator_description": "string or null",
    "primary_outcomes": [
        {
            "name": "outcome name",
            "measure_type": "one of the types above",
            "value": number,
            "ci_lower": number or null,
            "ci_upper": number or null,
            "p_value": number or null,
            "supporting_text": "exact quote from text"
        }
    ],
    "secondary_outcomes": [],
    "followup_weeks": number or null,
    "country": ["list", "of", "countries"] or null,
    "funding_source": "string or null",
    "limitations": ["list", "of", "limitations"] or null,
    "key_findings_text": "main findings summary",
    "supporting_snippets": [
        {
            "text": "exact quote",
            "location": "section name or page"
        }
    ]
}"""


class StudyCardExtractor:
    """
    Extract StudyCard from document text using LLM.

    Usage:
        extractor = StudyCardExtractor()
        card = extractor.extract(record_id, document_text)
    """

    def __init__(
        self,
        provider: "BaseLLMProvider | None" = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize extractor.

        Args:
            provider: LLM provider instance. If None, creates one using create_provider().
            model: Model name (used only if provider is None).
        """
        from cdr.llm.base import BaseLLMProvider

        if provider is not None and isinstance(provider, BaseLLMProvider):
            self._provider = provider
        else:
            # Fallback: create provider if not passed
            self._provider = create_provider(provider=None, model=model)
        self._tracer = get_tracer("cdr.extraction")
        self._metrics = get_cdr_metrics()

    def extract(
        self,
        record_id: str,
        text: str,
        title: str | None = None,
        pmid: str | None = None,
    ) -> StudyCard:
        """
        Extract StudyCard from text.

        Args:
            record_id: Record identifier.
            text: Document text.
            title: Study title.
            pmid: PubMed ID.

        Returns:
            Extracted StudyCard.
        """
        with self._tracer.span("extract_study_card", attributes={"record_id": record_id}) as span:
            # Truncate text if too long
            max_chars = 15000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"

            prompt = f"""## Study Text

Title: {title or "Unknown"}
PMID: {pmid or "Not available"}

---

{text}

---

Extract the structured study information from the above text."""

            messages = build_messages(
                system=EXTRACTION_SYSTEM_PROMPT,
                user=prompt,
            )

            try:
                response = self._provider.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=2000,
                )

                self._metrics.llm_requests.inc(labels={"operation": "extraction"})
                self._metrics.llm_tokens.inc(
                    response.total_tokens, labels={"operation": "extraction"}
                )

                card = self._parse_response(record_id, response.content, title, pmid)
                span.set_attribute(
                    "study_type", card.study_type.value if card.study_type else "unknown"
                )

                return card

            except Exception as e:
                span.set_status("error", str(e))
                raise ExtractionError(f"Failed to extract StudyCard: {e}") from e

    def _parse_response(
        self,
        record_id: str,
        content: str,
        title: str | None,
        pmid: str | None,
    ) -> StudyCard:
        """Parse LLM response into StudyCard."""
        # Multi-strategy JSON extraction
        data = self._extract_json(content)

        if data is None:
            # Fallback: create minimal StudyCard from title
            return StudyCard(
                record_id=record_id,
                study_type=StudyType.UNKNOWN,
                population_extracted=None,
                intervention_extracted=None,
                comparator_extracted=None,
            )

        # Parse study type with fallback to UNKNOWN
        study_type = StudyType.UNKNOWN
        if data.get("study_type"):
            try:
                # Normalize: lowercase, replace spaces/dashes with underscore
                st_raw = str(data["study_type"]).lower().replace("-", "_").replace(" ", "_")
                study_type = StudyType(st_raw)
            except ValueError:
                study_type = StudyType.UNKNOWN

        # Parse outcomes
        primary_outcomes = self._parse_outcomes(data.get("primary_outcomes", []))
        secondary_outcomes = self._parse_outcomes(data.get("secondary_outcomes", []))

        # Parse supporting snippets
        supporting_snippets: list[str] = []
        snippet_refs = data.get("supporting_snippets", [])
        for i, snip in enumerate(snippet_refs):
            snippet_id = f"{record_id}_snip_{i}"
            supporting_snippets.append(snippet_id)

        # Handle country as list or string
        country_raw = data.get("country")
        if isinstance(country_raw, list):
            country = ", ".join(str(c) for c in country_raw) if country_raw else None
        else:
            country = str(country_raw) if country_raw else None

        # Combine all outcomes
        all_outcomes = primary_outcomes + secondary_outcomes

        # CRITICAL: Never use placeholder snippet IDs - violates PRISMA trazability
        # If no snippets extracted, use empty list and let downstream handle it
        # Refs: PRISMA 2020, GRADE handbook
        return StudyCard(
            record_id=record_id,
            study_type=study_type,
            sample_size=data.get("population_n") or data.get("sample_size"),
            population_extracted=data.get("population_description"),
            intervention_extracted=data.get("intervention_description"),
            comparator_extracted=data.get("comparator_description"),
            primary_outcome=data.get("primary_outcome_name"),
            outcomes=all_outcomes,
            follow_up_duration=str(data.get("followup_weeks"))
            if data.get("followup_weeks")
            else None,
            country=country,
            funding_source=data.get("funding_source"),
            supporting_snippet_ids=supporting_snippets,  # Empty list if no snippets
        )

    def _parse_outcomes(self, outcomes_data: list[dict[str, Any]]) -> list[OutcomeMeasure]:
        """Parse outcome measures from data."""
        outcomes: list[OutcomeMeasure] = []

        for om in outcomes_data:
            try:
                measure_type = OutcomeMeasureType.OTHER
                if om.get("measure_type"):
                    try:
                        measure_type = OutcomeMeasureType(om["measure_type"])
                    except ValueError:
                        pass

                outcomes.append(
                    OutcomeMeasure(
                        name=om.get("name", "Unknown outcome"),
                        measure_type=measure_type,
                        value=om.get("value"),
                        ci_lower=om.get("ci_lower"),
                        ci_upper=om.get("ci_upper"),
                        p_value=om.get("p_value"),
                    )
                )
            except Exception:
                continue

        return outcomes

    def _extract_json(self, content: str) -> dict | None:
        """Multi-strategy JSON extraction from LLM response."""
        import re

        content = content.strip()

        # Strategy 1: Clean code blocks and try direct parse
        cleaned = content
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object with regex
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Strategy 3: Find largest {...} block
        start = content.find("{")
        if start != -1:
            depth = 0
            for i, c in enumerate(content[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break

        # Strategy 4: Try fixing common JSON issues
        if cleaned:
            # Fix trailing commas
            fixed = re.sub(r",\s*}", "}", cleaned)
            fixed = re.sub(r",\s*]", "]", fixed)
            # Fix unquoted keys
            fixed = re.sub(r"(\w+):", r'"\1":', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        return None


def extract_snippets(
    record_id: str,
    text: str,
    title: str | None = None,
    pmid: str | None = None,
) -> list[Snippet]:
    """
    Extract citable snippets from document text.

    Args:
        record_id: Record identifier.
        text: Document text.
        title: Study title.
        pmid: PubMed ID.

    Returns:
        List of Snippet objects.
    """
    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    snippets: list[Snippet] = []
    for i, para in enumerate(paragraphs):
        if len(para) < 50:  # Skip very short paragraphs
            continue

        snippet_id = f"{record_id}_snip_{i}"

        source_ref = SourceRef(
            record_id=record_id,
            pmid=pmid,
            title=title,
            section=None,  # Would need section detection
        )

        snippets.append(
            Snippet(
                snippet_id=snippet_id,
                text=para,
                source_ref=source_ref,
            )
        )

    return snippets

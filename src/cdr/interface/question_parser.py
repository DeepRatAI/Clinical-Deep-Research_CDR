"""
CDR Question Parser

Parse research questions into PICO framework components.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cdr.core.enums import ComparatorSource
from cdr.core.schemas import PICO
from cdr.observability.tracer import tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

QUESTION_PARSER_PROMPT = """You are an expert in evidence-based medicine and systematic review methodology.

Parse the research question into PICO components and return ONLY valid JSON, nothing else.

PICO Components:
- P (Population): Who are the patients/participants?
- I (Intervention): What is the treatment/intervention?
- C (Comparator): What is the comparison? 
- O (Outcome): What is the primary outcome?

CRITICAL RULES FOR COMPARATOR:
1. If the user explicitly mentions a comparator (e.g., "vs placebo", "compared to warfarin"), set:
   - comparator: the explicit comparator
   - comparator_source: "user_specified"

2. If the question asks about EFFICACY/EFFECTIVENESS without explicit comparator (e.g., "Does X prevent Y?", "Is X effective for Y?"), INFER:
   - comparator: "placebo or no treatment" (for efficacy questions)
   - comparator_source: "assumed_from_question"

3. If the question is HEAD-TO-HEAD (e.g., "X vs Y"), set:
   - comparator: the specific comparator drug/intervention
   - comparator_source: "user_specified"

4. If the question is truly non-comparative (e.g., "What is the mechanism of X?"), set:
   - comparator: null
   - comparator_source: "not_applicable"

You MUST respond with ONLY this JSON format, no explanations:
{
    "population": "...",
    "intervention": "...",
    "comparator": "...",
    "comparator_source": "user_specified|assumed_from_question|not_applicable",
    "outcome": "...",
    "study_types": ["RCT"]
}

Be specific. Never leave comparator null for efficacy questions - infer "placebo or no treatment" with source "assumed_from_question"."""


# =============================================================================
# PARSER CLASS
# =============================================================================


class QuestionParser:
    """Parse research questions into PICO framework."""

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        model: str = "gpt-4o",
    ) -> None:
        """Initialize parser.

        Args:
            llm_provider: LLM provider for parsing
            model: Model to use
        """
        self.llm = llm_provider
        self.model = model

    async def parse(self, question: str) -> PICO:
        """Parse a research question into PICO components.

        Args:
            question: The research question to parse

        Returns:
            PICO object with extracted components
        """
        with tracer.start_span("question_parser.parse") as span:
            span.set_attribute("question_length", len(question))

            messages = [
                {"role": "system", "content": QUESTION_PARSER_PROMPT},
                {"role": "user", "content": f"Parse this research question:\n\n{question}"},
            ]

            response = await self.llm.acomplete(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=1000,
            )

            pico = self._parse_response(response.content)

            span.set_attribute("outcome", pico.outcome)

            return pico

    def _parse_response(self, content: str) -> PICO:
        """Parse LLM response into PICO object."""
        import re

        original_content = content
        content = content.strip()

        # Try multiple extraction strategies
        data = None

        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from markdown code block
        if data is None and "```" in content:
            try:
                json_block = content.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                data = json.loads(json_block.strip())
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: Find JSON object in text using regex
        if data is None:
            json_match = re.search(r'\{[^{}]*"population"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        # Strategy 4: Parse nested JSON more aggressively
        if data is None:
            # Find anything that looks like JSON
            start = content.find("{")
            if start != -1:
                # Find matching closing brace
                depth = 0
                end = start
                for i, c in enumerate(content[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                try:
                    data = json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

        # If all strategies failed, fallback
        if data is None:
            print(f"[QuestionParser] All JSON extraction strategies failed")
            print(f"[QuestionParser] Raw response (first 300 chars): {original_content[:300]}")
            return PICO(
                population="Not parsed",
                intervention="Not parsed",
                outcome="Not specified",
            )

        # Handle both 'outcome' (singular) and 'outcomes' (plural) from LLM
        outcome = data.get("outcome")
        if not outcome and "outcomes" in data:
            outcomes = data.get("outcomes", [])
            outcome = outcomes[0] if outcomes else "Not specified"

        # Normalize study_types to lowercase for enum compatibility
        # CRITICAL: Filter out invalid study types to prevent Pydantic validation errors
        # Refs: https://errors.pydantic.dev/2.12/v/enum
        from cdr.core.enums import StudyType

        valid_study_types = {st.value for st in StudyType}

        study_types_raw = data.get("study_types", [])
        study_types = []
        if study_types_raw:
            for st in study_types_raw:
                if isinstance(st, str):
                    normalized = st.lower().replace("-", "_").replace(" ", "_")
                    # Map common aliases to valid enum values
                    alias_map = {
                        "cct": "rct",  # Controlled Clinical Trial -> RCT
                        "controlled_clinical_trial": "rct",
                        "randomized_controlled_trial": "rct",
                        "randomised_controlled_trial": "rct",
                        "observational": "cohort",
                        "prospective_cohort": "cohort",
                        "retrospective_cohort": "cohort",
                        "registry": "cohort",
                        "review": "narrative_review",
                        "literature_review": "narrative_review",
                    }
                    normalized = alias_map.get(normalized, normalized)
                    # Only add if it's a valid enum value
                    if normalized in valid_study_types:
                        study_types.append(normalized)
                    else:
                        print(
                            f"[QuestionParser] Skipping invalid study type: {st} (normalized: {normalized})"
                        )

        # Parse comparator_source from LLM response
        # CRITICAL: Must track how comparator was determined for semantic coherence
        # Refs: PRISMA 2020, WHO ICTRP, CDR DoD P1
        comparator_source_raw = data.get("comparator_source", "")
        comparator = data.get("comparator")

        # Map LLM response to enum values
        source_map = {
            "user_specified": ComparatorSource.USER_SPECIFIED,
            "assumed_from_question": ComparatorSource.ASSUMED_FROM_QUESTION,
            "inferred_from_evidence": ComparatorSource.INFERRED_FROM_EVIDENCE,
            "heuristic": ComparatorSource.HEURISTIC,
            "not_applicable": ComparatorSource.NOT_APPLICABLE,
        }

        if comparator_source_raw in source_map:
            comparator_source = source_map[comparator_source_raw]
        elif comparator:
            # LLM didn't specify source but gave comparator - default to assumed
            comparator_source = ComparatorSource.ASSUMED_FROM_QUESTION
        else:
            # No comparator - not applicable
            comparator_source = ComparatorSource.NOT_APPLICABLE

        return PICO(
            population=data.get("population", "Not specified"),
            intervention=data.get("intervention", "Not specified"),
            comparator=comparator,
            comparator_source=comparator_source,
            outcome=outcome or "Not specified",
            study_types=study_types,
        )

    def parse_sync(self, question: str) -> PICO:
        """Synchronous version of parse.

        Args:
            question: The research question to parse

        Returns:
            PICO object with extracted components
        """
        import asyncio

        return asyncio.run(self.parse(question))

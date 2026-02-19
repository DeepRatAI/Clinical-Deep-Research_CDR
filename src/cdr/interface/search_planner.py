"""
CDR Search Planner

Generate search strategies from PICO components.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cdr.core.schemas import PICO, SearchPlan
from cdr.observability.tracer import tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SEARCH_PLANNER_PROMPT = """You are a medical librarian. Generate a simple but effective PubMed search query.

IMPORTANT: Keep queries SIMPLE. PubMed works best with basic searches.
- Use simple terms without excessive MeSH qualifiers
- Combine main concepts with AND
- Use OR for synonyms sparingly
- Avoid overly specific qualifiers that reduce results

Example good query: "SGLT2 inhibitors AND heart failure AND mortality"
Example bad query: (complex query with many MeSH terms and filters)

Output ONLY this JSON format:
{
    "pubmed_query": "simple search terms",
    "ct_gov_query": "condition intervention keywords"
}"""


# =============================================================================
# PLANNER CLASS
# =============================================================================


class SearchPlanner:
    """Generate search strategies from PICO."""

    def __init__(
        self,
        llm_provider: "BaseLLMProvider",
        model: str = "gpt-4o",
    ) -> None:
        """Initialize planner.

        Args:
            llm_provider: LLM provider for planning
            model: Model to use
        """
        self.llm = llm_provider
        self.model = model

    async def plan(self, pico: PICO) -> SearchPlan:
        """Generate search plan from PICO.

        Args:
            pico: PICO framework components

        Returns:
            SearchPlan with queries for each database
        """
        with tracer.start_span("search_planner.plan") as span:
            pico_text = self._format_pico(pico)

            messages = [
                {"role": "system", "content": SEARCH_PLANNER_PROMPT},
                {"role": "user", "content": f"Generate search strategy for:\n\n{pico_text}"},
            ]

            response = await self.llm.acomplete(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2000,
            )

            print(f"[SearchPlanner] LLM response (first 500 chars): {response.content[:500]}")
            plan = self._parse_response(response.content, pico)
            print(f"[SearchPlanner] Generated PubMed query: {plan.pubmed_query}")
            print(f"[SearchPlanner] Generated CT.gov query: {plan.ct_gov_query}")

            span.set_attribute("pubmed_query_length", len(plan.pubmed_query or ""))

            return plan

    def _format_pico(self, pico: PICO) -> str:
        """Format PICO for prompt."""
        lines = [
            f"Population: {pico.population}",
            f"Intervention: {pico.intervention}",
            f"Comparator: {pico.comparator or 'Not specified'}",
            f"Outcome: {pico.outcome}",
        ]

        if pico.study_types:
            lines.append(f"Study types: {', '.join(str(st) for st in pico.study_types)}")

        return "\n".join(lines)

    def _parse_response(self, content: str, pico: PICO) -> SearchPlan:
        """Parse LLM response into SearchPlan."""
        import re

        original_content = content
        content = content.strip()

        # Try to extract JSON from various formats
        data = None

        # Strategy 1: Direct parse
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Code block
        if data is None and "```" in content:
            try:
                json_block = content.split("```")[1]
                if json_block.startswith("json"):
                    json_block = json_block[4:]
                data = json.loads(json_block.strip())
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: Find JSON object
        if data is None:
            start = content.find("{")
            if start != -1:
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

        if data is None:
            print(f"[SearchPlanner] JSON extraction failed, using fallback")
            return self._generate_fallback_plan(pico)

        # Parse date range
        date_range_data = data.get("date_range", {})
        date_range = None
        if (
            isinstance(date_range_data, dict)
            and date_range_data.get("start")
            and date_range_data.get("end")
        ):
            date_range = (date_range_data["start"], date_range_data["end"])

        return SearchPlan(
            pico=pico,
            pubmed_query=data.get("pubmed_query", ""),
            ct_gov_query=data.get("ct_gov_query"),
            date_range=date_range,
        )

    def _generate_fallback_plan(self, pico: PICO) -> SearchPlan:
        """Generate basic search plan when LLM fails."""
        # Build more sensible PubMed query with key terms only
        # Extract key terms from population and intervention

        # For SGLT2 inhibitors
        if "sglt2" in pico.intervention.lower():
            intervention_terms = "(SGLT2 inhibitor* OR sodium-glucose cotransporter-2 inhibitor* OR empagliflozin OR dapagliflozin OR canagliflozin)"
        else:
            # Use first few words
            intervention_words = pico.intervention.split()[:4]
            intervention_terms = " ".join(intervention_words)

        # For heart failure
        if "heart failure" in pico.population.lower():
            population_terms = '("heart failure"[MeSH] OR "heart failure"[tiab] OR HFrEF[tiab])'
        else:
            population_words = pico.population.split()[:4]
            population_terms = " ".join(population_words)

        # Outcome - simplified
        outcome_terms = "(mortality OR hospitalization OR quality of life)"

        pubmed_query = f"{intervention_terms} AND {population_terms} AND {outcome_terms}"

        # Basic ClinicalTrials.gov query (simpler is better)
        ct_query = (
            "SGLT2 inhibitor heart failure"
            if "sglt2" in pico.intervention.lower()
            else f"{intervention_words[0] if intervention_words else ''} {population_words[0] if population_words else ''}"
        )

        return SearchPlan(
            pico=pico,
            pubmed_query=pubmed_query,
            ct_gov_query=ct_query,
        )

    def plan_sync(self, pico: PICO) -> SearchPlan:
        """Synchronous version of plan.

        Args:
            pico: PICO framework components

        Returns:
            SearchPlan with queries
        """
        import asyncio

        return asyncio.run(self.plan(pico))


# =============================================================================
# QUERY VALIDATION
# =============================================================================


def validate_pubmed_query(query: str) -> dict:
    """Validate PubMed query syntax.

    Args:
        query: PubMed query string

    Returns:
        Dict with validation results
    """
    issues = []

    # Check for balanced brackets
    if query.count("(") != query.count(")"):
        issues.append("Unbalanced parentheses")

    if query.count("[") != query.count("]"):
        issues.append("Unbalanced square brackets")

    # Check for proper Boolean operators
    invalid_bools = ["and", "or", "not"]
    for term in invalid_bools:
        if f" {term} " in query.lower():
            if f" {term.upper()} " not in query:
                issues.append(f"Boolean operator '{term}' should be uppercase")

    # Check for common field tags
    valid_tags = ["tiab", "MeSH Terms", "MH", "pt", "la", "au", "dp"]
    import re

    tags_found = re.findall(r"\[([^\]]+)\]", query)
    for tag in tags_found:
        if not any(t in tag for t in valid_tags):
            # Not necessarily an issue, just a note
            pass

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "query_length": len(query),
        "estimated_complexity": query.count(" AND ") + query.count(" OR ") + 1,
    }

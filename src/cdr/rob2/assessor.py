"""
Risk of Bias 2 (RoB2) Assessor

Assessment of risk of bias for randomized trials using RoB2 framework.

Uses structured outputs via response_format when available for reliable JSON.
Documentation: https://methods.cochrane.org/bias/resources/rob-2-revised-cochrane-risk-bias-tool-randomized-trials
"""

import json
from typing import Any

from cdr.core.enums import RoB2Domain, RoB2Judgment
from cdr.core.schemas import RoB2DomainResult, RoB2Result
from cdr.core.exceptions import ExtractionError
from cdr.llm import create_provider, build_messages
from cdr.observability import get_tracer, get_cdr_metrics


ROB2_SYSTEM_PROMPT = """You are an expert in assessing risk of bias for randomized controlled trials using the Cochrane Risk of Bias 2 (RoB2) tool.

CRITICAL RULES:
1. Assess EACH of the 5 domains independently
2. For each domain, provide a judgment: LOW, SOME_CONCERNS, or HIGH
3. Provide a detailed rationale based on the study text
4. Quote supporting text snippets for each judgment
5. If information is insufficient, lean towards SOME_CONCERNS, not LOW

The 5 RoB2 Domains:

1. RANDOMIZATION (Bias arising from the randomization process)
   - Was the allocation sequence random?
   - Was the allocation sequence concealed?
   - Were there baseline imbalances?

2. DEVIATIONS (Bias due to deviations from intended interventions)
   - Were participants aware of their assigned intervention?
   - Were caregivers/staff aware of intervention?
   - Were there deviations due to the trial context?
   - Were appropriate analyses used?

3. MISSING_DATA (Bias due to missing outcome data)
   - Were outcome data available for all/nearly all participants?
   - Is there evidence that the result was not biased by missing data?

4. OUTCOME_MEASUREMENT (Bias in measurement of the outcome)
   - Was the method of measuring the outcome appropriate?
   - Could measurement differ between groups?
   - Were outcome assessors aware of intervention?

5. SELECTIVE_REPORTING (Bias in selection of the reported result)
   - Was the trial registered with pre-specified outcomes?
   - Were all pre-specified outcomes reported?
   - Were multiple analyses performed?

Output as JSON:
{
    "domains": [
        {
            "domain": "RANDOMIZATION",
            "judgment": "LOW" | "SOME_CONCERNS" | "HIGH",
            "rationale": "detailed explanation",
            "supporting_text": "quoted text from study"
        },
        ... (one for each of 5 domains)
    ]
}"""


class RoB2Assessor:
    """
    Assess risk of bias using RoB2 framework.

    Usage:
        assessor = RoB2Assessor()
        result = assessor.assess(record_id, study_text, study_card)
    """

    def __init__(
        self,
        provider: "BaseLLMProvider | None" = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize assessor.

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
        self._tracer = get_tracer("cdr.rob2")
        self._metrics = get_cdr_metrics()

    def assess(
        self,
        record_id: str,
        text: str,
        study_info: dict[str, Any] | None = None,
    ) -> RoB2Result:
        """
        Perform RoB2 assessment.

        Args:
            record_id: Record identifier.
            text: Study text (methods section preferred).
            study_info: Optional study metadata.

        Returns:
            RoB2Result with all domain assessments.
        """
        with self._tracer.span("assess_rob2", attributes={"record_id": record_id}) as span:
            # Truncate text if needed
            max_chars = 12000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"

            # Build context
            context = ""
            if study_info:
                context = f"""## Study Information
Study Type: {study_info.get("study_type", "Unknown")}
Population: {study_info.get("population_n", "Unknown")} participants
Intervention: {study_info.get("intervention_description", "Unknown")}
Comparator: {study_info.get("comparator_description", "Unknown")}

---

"""

            prompt = f"""{context}## Study Text for RoB2 Assessment

{text}

---

Assess the risk of bias for this randomized trial across all 5 RoB2 domains."""

            messages = build_messages(
                system=ROB2_SYSTEM_PROMPT,
                user=prompt,
            )

            try:
                # Try structured output first for reliable JSON
                # Import here to avoid circular imports
                try:
                    from cdr.llm.structured_outputs import get_rob2_response_format

                    response_format = get_rob2_response_format()
                except ImportError:
                    response_format = None

                response = self._provider.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=3000,
                    response_format=response_format,
                )

                self._metrics.llm_requests.inc(labels={"operation": "rob2"})
                self._metrics.llm_tokens.inc(response.total_tokens, labels={"operation": "rob2"})

                result = self._parse_response(record_id, response.content)
                span.set_attribute("overall_judgment", result.overall_judgment.value)

                return result

            except Exception as e:
                span.set_status("error", str(e))
                raise ExtractionError(f"RoB2 assessment failed: {e}") from e

    def _parse_response(self, record_id: str, content: str) -> RoB2Result:
        """Parse LLM response into RoB2Result."""
        import re

        # Clean JSON - handle various formats LLMs may produce
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        # Try to extract JSON from the content if it's embedded in text
        # Look for { ... } pattern
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # If still fails, log the content for debugging
            print(f"[RoB2] Failed to parse JSON. Content (first 200 chars): {content[:200]}")
            raise ExtractionError(f"Failed to parse RoB2 JSON: {e}")

        domains_data = data.get("domains", [])

        # Ensure all 5 domains are present
        domain_results: list[RoB2DomainResult] = []
        domains_found = set()

        for dom_data in domains_data:
            try:
                domain = RoB2Domain(dom_data.get("domain"))
                judgment = RoB2Judgment(dom_data.get("judgment", "SOME_CONCERNS"))

                domains_found.add(domain)

                domain_results.append(
                    RoB2DomainResult(
                        domain=domain,
                        judgment=judgment,
                        rationale=dom_data.get("rationale", "No rationale provided"),
                        supporting_snippet_ids=[f"{record_id}_rob2_{domain.value}"],
                    )
                )
            except (ValueError, KeyError):
                continue

        # Add missing domains with SOME_CONCERNS
        for domain in RoB2Domain:
            if domain not in domains_found:
                domain_results.append(
                    RoB2DomainResult(
                        domain=domain,
                        judgment=RoB2Judgment.SOME_CONCERNS,
                        rationale="Information not available in text",
                        supporting_snippet_ids=[f"{record_id}_rob2_{domain.value}"],
                    )
                )

        # Calculate overall judgment per RoB2 algorithm:
        # - HIGH if any domain is HIGH
        # - LOW if all domains are LOW
        # - SOME_CONCERNS otherwise
        judgments = [d.judgment for d in domain_results]
        if RoB2Judgment.HIGH in judgments:
            overall_judgment = RoB2Judgment.HIGH
            overall_rationale = "High risk of bias in at least one domain"
        elif all(j == RoB2Judgment.LOW for j in judgments):
            overall_judgment = RoB2Judgment.LOW
            overall_rationale = "Low risk of bias across all domains"
        else:
            overall_judgment = RoB2Judgment.SOME_CONCERNS
            concerns_domains = [
                d.domain.value for d in domain_results if d.judgment == RoB2Judgment.SOME_CONCERNS
            ]
            overall_rationale = f"Some concerns in domains: {', '.join(concerns_domains)}"

        return RoB2Result(
            record_id=record_id,
            domains=domain_results,
            overall_judgment=overall_judgment,
            overall_rationale=overall_rationale,
        )

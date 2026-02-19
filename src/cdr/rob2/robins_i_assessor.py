"""
ROBINS-I Assessor for Non-Randomized Studies

Assessment of risk of bias for non-randomized studies of interventions using ROBINS-I framework.

HIGH-3 fix: Proper bias assessment for observational studies.
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-3
Documentation: https://methods.cochrane.org/bias/resources/robins-i-tool
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from cdr.core.enums import ROBINSIDomain, ROBINSIJudgment
from cdr.core.exceptions import ExtractionError
from cdr.core.schemas import ROBINSIDomainResult, ROBINSIResult
from cdr.llm import build_messages, create_provider
from cdr.observability import SpanStatus, get_cdr_metrics, get_tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


ROBINS_I_SYSTEM_PROMPT = """You are an expert in assessing risk of bias for non-randomized studies of interventions using the ROBINS-I tool (Risk Of Bias In Non-randomized Studies of Interventions).

CRITICAL RULES:
1. ROBINS-I is specifically for NON-RANDOMIZED studies (cohort, case-control, cross-sectional)
2. Assess EACH of the 7 domains independently
3. Use the 5-level judgment scale: LOW, MODERATE, SERIOUS, CRITICAL, NO_INFORMATION
4. Provide detailed rationale based on the study text
5. Quote supporting text snippets for each judgment

The 7 ROBINS-I Domains:

1. CONFOUNDING (Bias due to confounding)
   - Did the study control for important confounders?
   - Were confounders measured validly and reliably?
   - Were confounders handled appropriately in analysis?
   
2. SELECTION (Bias in selection of participants into the study)
   - Was selection into the study based on participant characteristics observed after the start of intervention?
   - Did start of follow-up and start of intervention coincide?

3. CLASSIFICATION (Bias in classification of interventions)
   - Was intervention status well-defined?
   - Was information on intervention status recorded at the time of intervention?
   - Was intervention classification affected by knowledge of the outcome?

4. DEVIATIONS (Bias due to deviations from intended interventions)
   - Were there deviations from intended intervention beyond what would be expected in usual practice?
   - Were important co-interventions balanced?

5. MISSING_DATA (Bias due to missing data)
   - Were outcome data available for all, or nearly all, participants?
   - Were participants excluded due to missing data on intervention or outcome?

6. MEASUREMENT (Bias in measurement of outcomes)
   - Was the outcome measure appropriate?
   - Could outcome measurement have differed between intervention groups?
   - Were outcome assessors aware of intervention received?

7. SELECTION_REPORTED (Bias in selection of the reported result)
   - Were multiple outcome measurements available?
   - Were multiple analyses of the intervention-outcome relationship possible?
   - Is the reported result likely selected from multiple outcomes/analyses?

JUDGMENT SCALE:
- LOW: The study is comparable to a well-performed RCT
- MODERATE: The study provides sound evidence but not comparable to RCT
- SERIOUS: The study has important problems in this domain
- CRITICAL: The study is too problematic to provide useful evidence
- NO_INFORMATION: Insufficient information to make a judgment

Output as JSON:
{
    "domains": [
        {
            "domain": "bias_due_to_confounding",
            "judgment": "LOW" | "MODERATE" | "SERIOUS" | "CRITICAL" | "NO_INFORMATION",
            "rationale": "detailed explanation of judgment",
            "supporting_text": "quoted text from study supporting the judgment"
        },
        ... (one for each of 7 domains)
    ]
}"""


class ROBINSIAssessor:
    """
    Assess risk of bias using ROBINS-I framework for observational studies.

    HIGH-3 fix: Dedicated assessor for non-RCT studies.

    Usage:
        assessor = ROBINSIAssessor()
        result = assessor.assess(record_id, study_text, study_info)
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize assessor.

        Args:
            provider: LLM provider instance. If None, creates one using create_provider().
            model: Model name (used only if provider is None).
        """
        from cdr.llm.base import BaseLLMProvider

        if provider is not None:
            # Accept any provider that has a complete() method (duck typing)
            # This allows both real providers and mocks in tests
            if isinstance(provider, BaseLLMProvider) or hasattr(provider, "complete"):
                self._provider = provider
            else:
                # Fallback if provider doesn't look right
                self._provider = create_provider(provider=None, model=model)
        else:
            # No provider passed, create default
            self._provider = create_provider(provider=None, model=model)
        self._tracer = get_tracer("cdr.robinsi")
        self._metrics = get_cdr_metrics()

    def assess(
        self,
        record_id: str,
        text: str,
        study_info: dict[str, Any] | None = None,
    ) -> ROBINSIResult:
        """
        Perform ROBINS-I assessment.

        Args:
            record_id: Record identifier.
            text: Study text (methods/results sections preferred).
            study_info: Optional study metadata.

        Returns:
            ROBINSIResult with all domain assessments.
        """
        with self._tracer.span("assess_robinsi", attributes={"record_id": record_id}) as span:
            # Truncate text if needed
            max_chars = 12000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"

            # Build context with observational study-specific information
            context = ""
            if study_info:
                context = f"""## Study Information
Study Type: {study_info.get("study_type", "Observational (unknown subtype)")}
Population: {study_info.get("population_n", "Unknown")} participants
Exposure/Intervention: {study_info.get("intervention_description", "Unknown")}
Comparator: {study_info.get("comparator_description", "Unknown")}

**Note**: This is a NON-RANDOMIZED study. Assess using ROBINS-I criteria, not RoB2.

---

"""

            prompt = f"""{context}## Study Text for ROBINS-I Assessment

{text}

---

Assess the risk of bias for this non-randomized study across all 7 ROBINS-I domains.
Pay particular attention to confounding control and selection bias, which are typically
the most problematic domains in observational research."""

            messages = build_messages(
                system=ROBINS_I_SYSTEM_PROMPT,
                user=prompt,
            )

            try:
                # Use regular completion (no structured output for now)
                response = self._provider.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=4000,
                )

                self._metrics.llm_requests.inc(labels={"operation": "robinsi"})
                self._metrics.llm_tokens.inc(response.total_tokens, labels={"operation": "robinsi"})

                result = self._parse_response(record_id, response.content)
                span.set_attribute("overall_judgment", result.overall_judgment.value)

                return result

            except Exception as e:
                span.set_status(SpanStatus.ERROR, str(e))
                raise ExtractionError(f"ROBINS-I assessment failed: {e}") from e

    def _parse_response(self, record_id: str, content: str) -> ROBINSIResult:
        """Parse LLM response into ROBINSIResult."""
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
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[ROBINS-I] Failed to parse JSON. Content (first 200 chars): {content[:200]}")
            raise ExtractionError(f"Failed to parse ROBINS-I JSON: {e}") from e

        domains_data = data.get("domains", [])

        # Domain name mapping (handle various formats)
        domain_name_map = {
            "confounding": ROBINSIDomain.CONFOUNDING,
            "bias_due_to_confounding": ROBINSIDomain.CONFOUNDING,
            "selection": ROBINSIDomain.SELECTION,
            "bias_in_selection_of_participants": ROBINSIDomain.SELECTION,
            "classification": ROBINSIDomain.CLASSIFICATION,
            "bias_in_classification_of_interventions": ROBINSIDomain.CLASSIFICATION,
            "deviations": ROBINSIDomain.DEVIATIONS,
            "bias_due_to_deviations_from_intended_interventions": ROBINSIDomain.DEVIATIONS,
            "missing_data": ROBINSIDomain.MISSING_DATA,
            "bias_due_to_missing_data": ROBINSIDomain.MISSING_DATA,
            "measurement": ROBINSIDomain.MEASUREMENT,
            "bias_in_measurement_of_outcomes": ROBINSIDomain.MEASUREMENT,
            "selection_reported": ROBINSIDomain.SELECTION_REPORTED,
            "bias_in_selection_of_reported_result": ROBINSIDomain.SELECTION_REPORTED,
        }

        # Judgment mapping
        judgment_map = {
            "low": ROBINSIJudgment.LOW,
            "moderate": ROBINSIJudgment.MODERATE,
            "serious": ROBINSIJudgment.SERIOUS,
            "critical": ROBINSIJudgment.CRITICAL,
            "no_information": ROBINSIJudgment.NO_INFORMATION,
            # Handle uppercase
            "LOW": ROBINSIJudgment.LOW,
            "MODERATE": ROBINSIJudgment.MODERATE,
            "SERIOUS": ROBINSIJudgment.SERIOUS,
            "CRITICAL": ROBINSIJudgment.CRITICAL,
            "NO_INFORMATION": ROBINSIJudgment.NO_INFORMATION,
        }

        domain_results: list[ROBINSIDomainResult] = []
        domains_found = set()

        for dom_data in domains_data:
            try:
                domain_raw = dom_data.get("domain", "").lower().strip()
                domain = domain_name_map.get(domain_raw)

                if domain is None:
                    # Try direct enum value
                    try:
                        domain = ROBINSIDomain(domain_raw)
                    except ValueError:
                        continue

                judgment_raw = dom_data.get("judgment", "NO_INFORMATION")
                judgment = judgment_map.get(judgment_raw, ROBINSIJudgment.NO_INFORMATION)

                # Get rationale with minimum length enforcement
                rationale = dom_data.get("rationale", "No rationale provided")
                if len(rationale) < 10:
                    rationale = (
                        f"Assessment: {rationale}"
                        if rationale
                        else "No rationale provided for this domain"
                    )

                # Create domain result BEFORE adding to domains_found
                domain_result = ROBINSIDomainResult(
                    domain=domain,
                    judgment=judgment,
                    rationale=rationale,
                    supporting_snippet_ids=[f"{record_id}_robinsi_{domain.value}"],
                )
                domain_results.append(domain_result)
                domains_found.add(domain)

            except (ValueError, KeyError, Exception):
                # If anything fails, skip this domain - it will be added as NO_INFORMATION
                continue

        # Add missing domains with NO_INFORMATION
        for domain in ROBINSIDomain:
            if domain not in domains_found:
                domain_results.append(
                    ROBINSIDomainResult(
                        domain=domain,
                        judgment=ROBINSIJudgment.NO_INFORMATION,
                        rationale="Information not available in text",
                        supporting_snippet_ids=[f"{record_id}_robinsi_{domain.value}"],
                    )
                )

        # Calculate overall judgment per ROBINS-I algorithm:
        # - CRITICAL if any domain is CRITICAL
        # - SERIOUS if any domain is SERIOUS (and none CRITICAL)
        # - MODERATE if any domain is MODERATE (and none SERIOUS/CRITICAL)
        # - LOW only if all domains are LOW
        # - NO_INFORMATION if any domain is NO_INFORMATION and would otherwise be LOW
        judgments = [d.judgment for d in domain_results]

        if ROBINSIJudgment.CRITICAL in judgments:
            overall_judgment = ROBINSIJudgment.CRITICAL
            critical_domains = [
                d.domain.value for d in domain_results if d.judgment == ROBINSIJudgment.CRITICAL
            ]
            overall_rationale = f"Critical risk of bias in domains: {', '.join(critical_domains)}"

        elif ROBINSIJudgment.SERIOUS in judgments:
            overall_judgment = ROBINSIJudgment.SERIOUS
            serious_domains = [
                d.domain.value for d in domain_results if d.judgment == ROBINSIJudgment.SERIOUS
            ]
            overall_rationale = f"Serious risk of bias in domains: {', '.join(serious_domains)}"

        elif ROBINSIJudgment.MODERATE in judgments:
            overall_judgment = ROBINSIJudgment.MODERATE
            moderate_domains = [
                d.domain.value for d in domain_results if d.judgment == ROBINSIJudgment.MODERATE
            ]
            overall_rationale = f"Moderate risk of bias in domains: {', '.join(moderate_domains)}"

        elif ROBINSIJudgment.NO_INFORMATION in judgments:
            overall_judgment = ROBINSIJudgment.NO_INFORMATION
            ni_domains = [
                d.domain.value
                for d in domain_results
                if d.judgment == ROBINSIJudgment.NO_INFORMATION
            ]
            overall_rationale = f"Insufficient information in domains: {', '.join(ni_domains)}"

        else:
            # All LOW
            overall_judgment = ROBINSIJudgment.LOW
            overall_rationale = (
                "Low risk of bias across all domains - comparable to well-performed RCT"
            )

        return ROBINSIResult(
            record_id=record_id,
            domains=domain_results,
            overall_judgment=overall_judgment,
            overall_rationale=overall_rationale,
        )

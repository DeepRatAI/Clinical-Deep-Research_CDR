"""
Structured Output Schemas for LLM Responses

This module defines JSON schemas for forcing structured outputs from LLMs.
Using response_format with json_schema ensures the model returns valid JSON
that conforms to our data contracts.

Documentation:
- HF Router API: https://huggingface.co/docs/api-inference/tasks/chat-completion
- TGI Guidance: https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance
- Qwen Function Calling: https://qwen.readthedocs.io/en/latest/framework/function_call.html
"""

from typing import Any


# =============================================================================
# RoB2 ASSESSMENT SCHEMA
# =============================================================================

# CRITICAL: These values MUST match enums.py exactly (source of truth)
# RoB2Domain: randomization_process, deviations_from_intended_interventions,
#             missing_outcome_data, measurement_of_outcome, selection_of_reported_result
# RoB2Judgment: low, some_concerns, high

ROB2_DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {
            "type": "string",
            "enum": [
                "randomization_process",
                "deviations_from_intended_interventions",
                "missing_outcome_data",
                "measurement_of_outcome",
                "selection_of_reported_result",
            ],
            "description": "The RoB2 domain being assessed (must use official Cochrane domain names)",
        },
        "judgment": {
            "type": "string",
            "enum": ["low", "some_concerns", "high"],
            "description": "Risk of bias judgment for this domain (lowercase)",
        },
        "rationale": {
            "type": "string",
            "minLength": 10,
            "description": "Detailed explanation of the judgment",
        },
        "supporting_text": {
            "type": "string",
            "description": "Quoted text from study supporting this judgment",
        },
    },
    "required": ["domain", "judgment", "rationale"],
}

ROB2_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": ROB2_DOMAIN_SCHEMA,
            "minItems": 5,
            "maxItems": 5,
            "description": "Assessment for all 5 RoB2 domains",
        }
    },
    "required": ["domains"],
}


def get_rob2_response_format() -> dict[str, Any]:
    """Get response_format dict for RoB2 structured output.

    Returns:
        Dict suitable for passing to HF Router API as response_format
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "RoB2Assessment",
            "description": "Risk of Bias 2 assessment for a randomized controlled trial",
            "strict": True,
            "schema": ROB2_ASSESSMENT_SCHEMA,
        },
    }


def get_rob2_tool() -> dict[str, Any]:
    """Get tool definition for RoB2 function calling.

    Returns:
        Tool definition for OpenAI-compatible function calling
    """
    return {
        "type": "function",
        "function": {
            "name": "submit_rob2_assessment",
            "description": "Submit the Risk of Bias 2 assessment for the study",
            "parameters": ROB2_ASSESSMENT_SCHEMA,
        },
    }


# =============================================================================
# SYNTHESIS / CLAIMS SCHEMA
# =============================================================================

# CRITICAL: These field names MUST match schemas.py EvidenceClaim (source of truth)
# Required fields: claim_id, claim_text, certainty, supporting_snippet_ids

EVIDENCE_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_id": {
            "type": "string",
            "description": "Unique identifier for this claim (e.g., claim_001)",
        },
        "claim_text": {
            "type": "string",
            "minLength": 20,
            "description": "Clear, concise statement of the finding (min 20 chars)",
        },
        "certainty": {
            "type": "string",
            "enum": ["high", "moderate", "low", "very_low"],
            "description": "GRADE certainty of evidence (lowercase)",
        },
        "supporting_snippet_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "IDs of snippets supporting this claim (REQUIRED, at least 1)",
        },
        "conflicting_snippet_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of snippets that conflict with this claim",
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Known limitations of this claim",
        },
        "studies_supporting": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of studies supporting this claim",
        },
        "studies_conflicting": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of studies conflicting with this claim",
        },
        "grade_rationale": {
            "type": "object",
            "properties": {
                "risk_of_bias": {"type": "string"},
                "inconsistency": {"type": "string"},
                "indirectness": {"type": "string"},
                "imprecision": {"type": "string"},
                "publication_bias": {"type": "string"},
            },
            "description": "GRADE assessment rationale",
        },
    },
    "required": ["claim_id", "claim_text", "certainty", "supporting_snippet_ids"],
}

SYNTHESIS_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": EVIDENCE_CLAIM_SCHEMA,
            "description": "Evidence claims with GRADE assessments",
        },
        "heterogeneity_assessment": {
            "type": "object",
            "properties": {
                "clinical": {"type": "string"},
                "methodological": {"type": "string"},
                "statistical": {"type": "string"},
            },
            "description": "Assessment of heterogeneity across studies",
        },
        "overall_synthesis": {
            "type": "string",
            "description": "Narrative summary of synthesized evidence",
        },
    },
    "required": ["claims"],
}


def get_synthesis_response_format() -> dict[str, Any]:
    """Get response_format dict for synthesis structured output.

    Returns:
        Dict suitable for passing to HF Router API as response_format
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "EvidenceSynthesis",
            "description": "Evidence synthesis with claims and GRADE assessments",
            "strict": True,
            "schema": SYNTHESIS_RESULT_SCHEMA,
        },
    }


def get_synthesis_tool() -> dict[str, Any]:
    """Get tool definition for synthesis function calling.

    Returns:
        Tool definition for OpenAI-compatible function calling
    """
    return {
        "type": "function",
        "function": {
            "name": "submit_evidence_synthesis",
            "description": "Submit the synthesized evidence claims with GRADE assessments",
            "parameters": SYNTHESIS_RESULT_SCHEMA,
        },
    }


# =============================================================================
# STUDY CARD EXTRACTION SCHEMA
# =============================================================================

STUDY_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "study_type": {
            "type": "string",
            "enum": [
                "RCT",
                "OBSERVATIONAL",
                "COHORT",
                "CASE_CONTROL",
                "CROSS_SECTIONAL",
                "CASE_SERIES",
                "CASE_REPORT",
                "SYSTEMATIC_REVIEW",
                "META_ANALYSIS",
                "UNKNOWN",
            ],
            "description": "Type of study design",
        },
        "sample_size": {"type": "integer", "minimum": 0, "description": "Number of participants"},
        "population_description": {
            "type": "string",
            "description": "Description of study population",
        },
        "intervention_description": {
            "type": "string",
            "description": "Description of intervention",
        },
        "comparator_description": {
            "type": "string",
            "description": "Description of comparator/control",
        },
        "primary_outcome": {"type": "string", "description": "Primary outcome measure"},
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key findings from the study",
        },
        "limitations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Study limitations",
        },
    },
    "required": ["study_type"],
}


def get_study_card_response_format() -> dict[str, Any]:
    """Get response_format dict for study card extraction.

    Returns:
        Dict suitable for passing to HF Router API as response_format
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "StudyCardExtraction",
            "description": "Structured data extraction from a clinical study",
            "strict": True,
            "schema": STUDY_CARD_SCHEMA,
        },
    }


def get_study_card_tool() -> dict[str, Any]:
    """Get tool definition for study card extraction function calling.

    Returns:
        Tool definition for OpenAI-compatible function calling
    """
    return {
        "type": "function",
        "function": {
            "name": "submit_study_card",
            "description": "Submit the extracted study card data",
            "parameters": STUDY_CARD_SCHEMA,
        },
    }

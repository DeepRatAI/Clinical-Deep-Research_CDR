"""
CDR Evaluation Module

RAGAs-inspired and CDR-specific quality metrics for clinical deep research.

Exports:
    - CDRMetricsEvaluator: Main evaluator class
    - EvaluationReport: Structured report output
    - MetricResult: Individual metric result
    - MetricStatus: Pass/Warn/Fail status
    - evaluate_cdr_output: Convenience function
    - GOLDEN_SET: Benchmark clinical questions
    - get_golden_set: Function to retrieve golden set
    - validate_against_golden_set: Validate results against expectations

Metrics implemented:
    Core (DoD thresholds):
        - snippet_coverage: Claims with supporting snippets
        - verification_coverage: Verified claims percentage
        - claims_evidence_ratio: Claims to snippets ratio
        - composition_emitted_rate: Hypothesis generation rate

    RAGAs-inspired:
        - context_precision: Retrieved context usage
        - answer_faithfulness: Claim grounding
        - citation_accuracy: Reference validity
"""

from cdr.evaluation.metrics import (
    CDRMetricsEvaluator,
    EvaluationReport,
    MetricResult,
    MetricStatus,
    evaluate_cdr_output,
)

from cdr.evaluation.golden_set import (
    GOLDEN_SET,
    GoldenSetQuestion,
    ExpectedEvidenceLevel,
    get_golden_set,
    get_question_by_id,
    validate_against_golden_set,
)

__all__ = [
    # Metrics
    "CDRMetricsEvaluator",
    "EvaluationReport",
    "MetricResult",
    "MetricStatus",
    "evaluate_cdr_output",
    # Golden Set
    "GOLDEN_SET",
    "GoldenSetQuestion",
    "ExpectedEvidenceLevel",
    "get_golden_set",
    "get_question_by_id",
    "validate_against_golden_set",
]

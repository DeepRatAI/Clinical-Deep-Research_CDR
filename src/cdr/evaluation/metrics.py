"""
CDR Evaluation Metrics

Implements quality metrics for Clinical Deep Research outputs.

Metrics defined per borrador arquitectónico:
- snippet_coverage >= 1.00 (DoD 2+): Every claim must have supporting snippets
- verification_coverage >= 0.95 (DoD 3): 95%+ claims verified against sources
- claims_with_evidence_ratio <= 1.2 (DoD 3): Minimal claims without direct evidence
- composition_emitted_rate >= 0.10 (DoD 3): At least 10% of runs produce hypotheses

Additional RAGAs-inspired metrics:
- context_precision: Proportion of retrieved context that's relevant
- answer_faithfulness: Degree to which claims are grounded in snippets
- citation_accuracy: Correctness of source references

References:
- RAGAs: https://docs.ragas.io/en/stable/concepts/
- CDR Architecture: deepresearch_ideasydata.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cdr.core.schemas import EvidenceClaim, Snippet, VerificationResult
from cdr.composition.schemas import ComposedHypothesis


class MetricStatus(str, Enum):
    """Status of a metric evaluation."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"


@dataclass
class MetricResult:
    """Result of evaluating a single metric."""

    name: str
    value: float
    threshold: float
    status: MetricStatus
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "status": self.status.value,
            "details": self.details,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report for a CDR run."""

    run_id: str
    dod_level: int
    metrics: list[MetricResult] = field(default_factory=list)
    overall_pass: bool = True
    summary: str = ""

    def add_metric(self, metric: MetricResult) -> None:
        """Add a metric result."""
        self.metrics.append(metric)
        if metric.status == MetricStatus.FAIL:
            self.overall_pass = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "dod_level": self.dod_level,
            "overall_pass": self.overall_pass,
            "summary": self.summary,
            "metrics": [m.to_dict() for m in self.metrics],
        }


class CDRMetricsEvaluator:
    """
    Evaluator for CDR output quality metrics.

    Implements the quality gates defined in the architecture document.
    """

    # DoD level thresholds
    DOD_THRESHOLDS = {
        2: {
            "snippet_coverage": 1.0,  # Every claim has snippets
            "verification_coverage": 0.80,  # 80%+ verified
        },
        3: {
            "snippet_coverage": 1.0,
            "verification_coverage": 0.95,  # 95%+ verified
            "claims_with_evidence_ratio": 1.2,  # Max ratio
            "composition_emitted_rate": 0.10,  # 10%+ emit hypotheses
        },
    }

    def __init__(self, dod_level: int = 3):
        """
        Initialize evaluator.

        Args:
            dod_level: Definition of Done level (1, 2, or 3)
        """
        self.dod_level = min(max(dod_level, 1), 3)
        self.thresholds = self.DOD_THRESHOLDS.get(self.dod_level, {})

    def evaluate_snippet_coverage(
        self,
        claims: list[EvidenceClaim],
    ) -> MetricResult:
        """
        Evaluate snippet coverage: every claim should have supporting snippets.

        Args:
            claims: List of evidence claims

        Returns:
            MetricResult with coverage score
        """
        threshold = self.thresholds.get("snippet_coverage", 1.0)

        if not claims:
            return MetricResult(
                name="snippet_coverage",
                value=1.0,
                threshold=threshold,
                status=MetricStatus.NOT_APPLICABLE,
                details="No claims to evaluate",
            )

        claims_with_snippets = sum(1 for c in claims if c.supporting_snippet_ids)
        coverage = claims_with_snippets / len(claims)

        status = MetricStatus.PASS if coverage >= threshold else MetricStatus.FAIL

        return MetricResult(
            name="snippet_coverage",
            value=coverage,
            threshold=threshold,
            status=status,
            details=f"{claims_with_snippets}/{len(claims)} claims have supporting snippets",
        )

    def evaluate_verification_coverage(
        self,
        claims: list[EvidenceClaim],
        verification_results: dict[str, VerificationResult] | None = None,
    ) -> MetricResult:
        """
        Evaluate verification coverage: claims verified against sources.

        Args:
            claims: List of evidence claims
            verification_results: Dict mapping claim_id to VerificationResult

        Returns:
            MetricResult with verification coverage
        """
        threshold = self.thresholds.get("verification_coverage", 0.80)

        if not claims:
            return MetricResult(
                name="verification_coverage",
                value=1.0,
                threshold=threshold,
                status=MetricStatus.NOT_APPLICABLE,
                details="No claims to evaluate",
            )

        if verification_results is None:
            # No verification data - can't evaluate
            return MetricResult(
                name="verification_coverage",
                value=0.0,
                threshold=threshold,
                status=MetricStatus.WARN,
                details="No verification results provided",
            )

        verified = sum(
            1
            for c in claims
            if c.claim_id in verification_results and verification_results[c.claim_id].passed
        )
        coverage = verified / len(claims)

        if coverage >= threshold:
            status = MetricStatus.PASS
        elif coverage >= threshold * 0.8:  # Within 20% of threshold
            status = MetricStatus.WARN
        else:
            status = MetricStatus.FAIL

        return MetricResult(
            name="verification_coverage",
            value=coverage,
            threshold=threshold,
            status=status,
            details=f"{verified}/{len(claims)} claims verified",
        )

    def evaluate_claims_evidence_ratio(
        self,
        claims: list[EvidenceClaim],
        snippets: list[Snippet],
    ) -> MetricResult:
        """
        Evaluate claims-to-evidence ratio.

        A ratio > 1.2 suggests claims are being made without adequate evidence.

        Args:
            claims: List of evidence claims
            snippets: List of source snippets

        Returns:
            MetricResult with ratio
        """
        threshold = self.thresholds.get("claims_with_evidence_ratio", 1.2)

        if not snippets:
            return MetricResult(
                name="claims_evidence_ratio",
                value=float("inf") if claims else 1.0,
                threshold=threshold,
                status=MetricStatus.FAIL if claims else MetricStatus.NOT_APPLICABLE,
                details="No snippets to evaluate against",
            )

        ratio = len(claims) / len(snippets) if snippets else float("inf")

        status = MetricStatus.PASS if ratio <= threshold else MetricStatus.FAIL

        return MetricResult(
            name="claims_evidence_ratio",
            value=ratio,
            threshold=threshold,
            status=status,
            details=f"{len(claims)} claims / {len(snippets)} snippets = {ratio:.2f}",
        )

    def evaluate_composition_rate(
        self,
        hypotheses: list[ComposedHypothesis],
        total_runs: int = 1,
    ) -> MetricResult:
        """
        Evaluate composition emission rate.

        At DoD 3, at least 10% of runs should produce hypotheses when evidence allows.

        Args:
            hypotheses: List of composed hypotheses
            total_runs: Number of runs evaluated (for batch)

        Returns:
            MetricResult with emission rate
        """
        threshold = self.thresholds.get("composition_emitted_rate", 0.10)

        if total_runs == 0:
            return MetricResult(
                name="composition_emitted_rate",
                value=0.0,
                threshold=threshold,
                status=MetricStatus.NOT_APPLICABLE,
                details="No runs to evaluate",
            )

        # For single run: 1 if hypotheses exist, 0 otherwise
        runs_with_hypotheses = 1 if hypotheses else 0
        rate = runs_with_hypotheses / total_runs

        status = MetricStatus.PASS if rate >= threshold else MetricStatus.WARN

        return MetricResult(
            name="composition_emitted_rate",
            value=rate,
            threshold=threshold,
            status=status,
            details=f"{len(hypotheses)} hypotheses generated",
        )

    def evaluate_context_precision(
        self,
        snippets: list[Snippet],
        claims: list[EvidenceClaim],
    ) -> MetricResult:
        """
        Evaluate context precision: proportion of snippets used by claims.

        Higher precision = less noise in retrieved context.

        Args:
            snippets: Retrieved source snippets
            claims: Generated claims

        Returns:
            MetricResult with precision score
        """
        if not snippets:
            return MetricResult(
                name="context_precision",
                value=1.0,
                threshold=0.7,
                status=MetricStatus.NOT_APPLICABLE,
                details="No snippets to evaluate",
            )

        # Count unique snippets referenced by claims
        referenced_ids = set()
        for claim in claims:
            referenced_ids.update(claim.supporting_snippet_ids)

        snippet_ids = {s.snippet_id for s in snippets}
        used = len(referenced_ids & snippet_ids)
        precision = used / len(snippets)

        threshold = 0.7  # 70% of retrieved context should be used
        status = MetricStatus.PASS if precision >= threshold else MetricStatus.WARN

        return MetricResult(
            name="context_precision",
            value=precision,
            threshold=threshold,
            status=status,
            details=f"{used}/{len(snippets)} snippets referenced by claims",
        )

    def evaluate_answer_faithfulness(
        self,
        claims: list[EvidenceClaim],
        verification_results: dict[str, VerificationResult] | None = None,
    ) -> MetricResult:
        """
        Evaluate answer faithfulness: claims grounded in evidence.

        Measures how well claims are supported by their referenced snippets.

        Args:
            claims: Evidence claims
            verification_results: Dict mapping claim_id to VerificationResult

        Returns:
            MetricResult with faithfulness score
        """
        if not claims:
            return MetricResult(
                name="answer_faithfulness",
                value=1.0,
                threshold=0.8,
                status=MetricStatus.NOT_APPLICABLE,
                details="No claims to evaluate",
            )

        # Faithfulness = claims with snippets that passed verification
        def is_grounded(c: EvidenceClaim) -> bool:
            has_snippets = bool(c.supporting_snippet_ids)
            if verification_results and c.claim_id in verification_results:
                return has_snippets and verification_results[c.claim_id].passed
            return has_snippets  # If no verification, at least has snippets

        grounded = sum(1 for c in claims if is_grounded(c))
        faithfulness = grounded / len(claims)

        threshold = 0.8
        status = MetricStatus.PASS if faithfulness >= threshold else MetricStatus.WARN

        return MetricResult(
            name="answer_faithfulness",
            value=faithfulness,
            threshold=threshold,
            status=status,
            details=f"{grounded}/{len(claims)} claims grounded in evidence",
        )

    def evaluate_citation_accuracy(
        self,
        claims: list[EvidenceClaim],
        snippets: list[Snippet],
    ) -> MetricResult:
        """
        Evaluate citation accuracy: all referenced snippets exist.

        Args:
            claims: Evidence claims
            snippets: Available snippets

        Returns:
            MetricResult with accuracy score
        """
        if not claims:
            return MetricResult(
                name="citation_accuracy",
                value=1.0,
                threshold=1.0,
                status=MetricStatus.NOT_APPLICABLE,
                details="No claims to evaluate",
            )

        snippet_ids = {s.snippet_id for s in snippets}

        valid_refs = 0
        total_refs = 0

        for claim in claims:
            for ref in claim.supporting_snippet_ids:
                total_refs += 1
                if ref in snippet_ids:
                    valid_refs += 1

        if total_refs == 0:
            return MetricResult(
                name="citation_accuracy",
                value=0.0,
                threshold=1.0,
                status=MetricStatus.WARN,
                details="No citations to evaluate",
            )

        accuracy = valid_refs / total_refs
        status = MetricStatus.PASS if accuracy == 1.0 else MetricStatus.FAIL

        return MetricResult(
            name="citation_accuracy",
            value=accuracy,
            threshold=1.0,
            status=status,
            details=f"{valid_refs}/{total_refs} citations valid",
        )

    def evaluate_run(
        self,
        run_id: str,
        claims: list[EvidenceClaim],
        snippets: list[Snippet],
        hypotheses: list[ComposedHypothesis] | None = None,
        verification_results: dict[str, VerificationResult] | None = None,
    ) -> EvaluationReport:
        """
        Run full evaluation on a CDR run output.

        Args:
            run_id: Identifier for the run
            claims: Generated evidence claims
            snippets: Source snippets
            hypotheses: Composed hypotheses (optional)
            verification_results: Dict mapping claim_id to VerificationResult

        Returns:
            Complete evaluation report
        """
        report = EvaluationReport(
            run_id=run_id,
            dod_level=self.dod_level,
        )

        # Core metrics (always evaluated)
        report.add_metric(self.evaluate_snippet_coverage(claims))
        report.add_metric(self.evaluate_verification_coverage(claims, verification_results))

        # DoD 3 metrics
        if self.dod_level >= 3:
            report.add_metric(self.evaluate_claims_evidence_ratio(claims, snippets))
            report.add_metric(self.evaluate_composition_rate(hypotheses or [], 1))

        # RAGAs-inspired metrics
        report.add_metric(self.evaluate_context_precision(snippets, claims))
        report.add_metric(self.evaluate_answer_faithfulness(claims, verification_results))
        report.add_metric(self.evaluate_citation_accuracy(claims, snippets))

        # Generate summary
        passed = [m for m in report.metrics if m.status == MetricStatus.PASS]
        failed = [m for m in report.metrics if m.status == MetricStatus.FAIL]
        warned = [m for m in report.metrics if m.status == MetricStatus.WARN]

        report.summary = (
            f"DoD {self.dod_level}: {len(passed)} passed, "
            f"{len(warned)} warnings, {len(failed)} failed"
        )

        return report


def evaluate_cdr_output(
    run_id: str,
    claims: list[EvidenceClaim],
    snippets: list[Snippet],
    hypotheses: list[ComposedHypothesis] | None = None,
    verification_results: dict[str, VerificationResult] | None = None,
    dod_level: int = 3,
) -> EvaluationReport:
    """
    Convenience function to evaluate CDR output.

    Args:
        run_id: Run identifier
        claims: Generated claims
        snippets: Source snippets
        hypotheses: Composed hypotheses
        verification_results: Dict mapping claim_id to VerificationResult
        dod_level: Definition of Done level

    Returns:
        Evaluation report
    """
    evaluator = CDRMetricsEvaluator(dod_level=dod_level)
    return evaluator.evaluate_run(run_id, claims, snippets, hypotheses, verification_results)

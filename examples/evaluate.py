#!/usr/bin/env python3
"""
Example: Evaluate CDR against Golden Set

Runs the evaluation framework against the built-in golden set
and prints a summary of metrics.

Requirements:
    pip install -e ".[dev]"

Usage:
    PYTHONPATH=src python examples/evaluate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdr.evaluation.golden_set import get_golden_set
from cdr.evaluation.metrics import CDRMetricsEvaluator


def main():
    """Show golden set questions and evaluation thresholds."""

    golden_set = get_golden_set()
    evaluator = CDRMetricsEvaluator(dod_level=3)

    print("📊  CDR Golden Set — Evaluation Overview")
    print("=" * 60)
    print()

    for q in golden_set:
        print(f"  {q.id}: {q.question[:60]}...")
        print(f"    Population:    {q.population[:50]}")
        print(f"    Intervention:  {q.intervention[:50]}")
        print(f"    Evidence:      {q.expected_evidence_level.value}")
        print(f"    Min studies:   {q.expected_min_studies}")
        print(f"    Composition:   {'Yes' if q.composition_expected else 'No'}")
        print()

    print("Quality Thresholds (DoD Level 3):")
    print("-" * 40)
    for metric, threshold in evaluator.thresholds.items():
        print(f"  {metric}: {threshold}")
    print()

    print(f"Total questions: {len(golden_set)}")
    print()
    print("Run full evaluation with:")
    print("  make eval")
    print("  # or")
    print("  PYTHONPATH=src python -m eval.eval_runner")


if __name__ == "__main__":
    main()

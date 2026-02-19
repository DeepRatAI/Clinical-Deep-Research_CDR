#!/usr/bin/env python3
"""
Example: Inspect a CDR Report

Loads a pre-generated CDR report JSON and prints a structured
summary of findings, PRISMA flow, and quality metrics.

This example requires NO API keys — it works entirely offline
with the bundled sample output.

Requirements:
    pip install -e ".[dev]"

Usage:
    python examples/inspect_report.py
    python examples/inspect_report.py --report path/to/custom_report.json
"""

import argparse
import json
import sys
from pathlib import Path

# Default to the bundled sample report
DEFAULT_REPORT = Path(__file__).resolve().parent / "output" / "sample_report.json"


def inspect_report(report_path: Path) -> None:
    """Load and print a structured summary of a CDR report."""
    if not report_path.exists():
        print(f"❌  Report not found: {report_path}")
        print("    Run `PYTHONPATH=src python examples/run_query.py` to generate one,")
        print("    or use the bundled sample: examples/output/sample_report.json")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    # ── Header ──────────────────────────────────────────────
    print("=" * 65)
    print("  CDR Report Inspector")
    print("=" * 65)
    print()

    question = report.get("question", report.get("clinical_question", "N/A"))
    print(f"  Question:  {question}")
    print(f"  Status:    {report.get('status', 'unknown')}")
    print(f"  Timestamp: {report.get('timestamp', 'N/A')}")
    print()

    # ── Disclaimer ──────────────────────────────────────────
    disclaimer = report.get("disclaimer")
    if disclaimer:
        print(f"  ⚠️  {disclaimer}")
        print()

    # ── PRISMA Flow ─────────────────────────────────────────
    prisma = report.get("prisma_counts", {})
    if prisma:
        print("  📊 PRISMA Flow")
        print("  " + "-" * 40)
        print(f"    Records identified : {prisma.get('records_identified', 0)}")
        print(f"    Records screened   : {prisma.get('records_screened', 0)}")
        print(f"    Full-text assessed : {prisma.get('full_text_assessed', 0)}")
        print(f"    Studies included   : {prisma.get('studies_included', 0)}")
        print()

    # ── Evidence Claims ─────────────────────────────────────
    claims = report.get("claims", [])
    if claims:
        print(f"  📋 Evidence Claims ({len(claims)})")
        print("  " + "-" * 40)
        for i, claim in enumerate(claims, 1):
            certainty = claim.get("certainty", "unknown").upper()
            text = claim.get("claim_text", "")[:100]
            snippet_ids = claim.get("supporting_snippet_ids", [])
            verified = claim.get("verified", None)
            v_str = "✅" if verified else ("❌" if verified is False else "?")
            print(f"    {i}. [{certainty}] {text}...")
            print(f"       Snippets: {len(snippet_ids)}  |  Verified: {v_str}")
        print()

    # ── Quality KPIs ────────────────────────────────────────
    kpis = report.get("run_kpis", {})
    if kpis:
        print("  🎯 Quality KPIs")
        print("  " + "-" * 40)
        for k, v in kpis.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f"    {label:30s}: {v:.2%}")
            else:
                print(f"    {label:30s}: {v}")
        print()

    # ── Composition (if any) ────────────────────────────────
    hypos = report.get("composed_hypotheses", [])
    if hypos:
        print(f"  💡 Composed Hypotheses ({len(hypos)})")
        print("  " + "-" * 40)
        for h in hypos:
            print(f"    • {h.get('hypothesis', h)}")
        print()

    # ── Stats ───────────────────────────────────────────────
    print("  📈 Summary Statistics")
    print("  " + "-" * 40)
    print(
        f"    Studies found     : {report.get('study_count', len(report.get('study_cards', [])))}"
    )
    print(f"    Claims generated  : {report.get('claim_count', len(claims))}")
    print(f"    Snippets extracted: {report.get('snippet_count', 0)}")
    print()

    # ── Schema validation ───────────────────────────────────
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "report.schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # type: ignore[import-untyped]

            with open(schema_path) as sf:
                schema = json.load(sf)
            jsonschema.validate(report, schema)
            print("  ✅ Report passes JSON Schema validation")
        except ImportError:
            print("  ℹ️  Install jsonschema to validate: pip install jsonschema")
        except jsonschema.ValidationError as ve:
            print(f"  ⚠️  Schema validation error: {ve.message}")
    else:
        print("  ℹ️  Schema not found at schemas/report.schema.json")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a CDR report JSON")
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help=f"Path to report JSON (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args()

    inspect_report(Path(args.report))


if __name__ == "__main__":
    main()

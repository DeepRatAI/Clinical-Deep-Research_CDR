#!/usr/bin/env python3
"""
CDR Online Demo — Execute real pipeline runs with LLM providers.

Produces:
    examples/output/online/run_01/sample_report_online.json
    examples/output/online/run_02/sample_report_online.json
    ...
    examples/output/online/run_05/sample_report_online.json

Requires .env to be configured with at least one LLM provider API key.

This is the script behind `make demo-online`.

Usage:
    PYTHONPATH=src python scripts/run_online_demo.py

Environment:
    Keys are read from .env / environment variables.
    NO keys are written to any output file.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Clinical questions for the 5 runs (from golden set)
DEMO_QUESTIONS = [
    {
        "id": "run_01",
        "question": "Is aspirin effective for secondary prevention of cardiovascular events?",
    },
    {
        "id": "run_02",
        "question": "Does metformin combined with GLP-1 agonists improve outcomes beyond monotherapy in type 2 diabetes?",
    },
    {
        "id": "run_03",
        "question": "Is vitamin D supplementation effective for preventing respiratory infections?",
    },
    {
        "id": "run_04",
        "question": "What is the accuracy of high-sensitivity troponin for diagnosing acute myocardial infarction?",
    },
    {
        "id": "run_05",
        "question": "Can anti-inflammatory therapies reduce cardiovascular events in patients with elevated CRP?",
    },
]

# Provider fallback order
PROVIDER_ORDER = [
    "openrouter",
    "groq",
    "cerebras",
    "gemini",
    "huggingface",
]

# Max records per source (keep low for demo speed)
MAX_RESULTS = 15


def get_working_provider():
    """Find the first provider with a configured API key."""
    from cdr.llm.factory import create_provider

    for provider_name in PROVIDER_ORDER:
        try:
            provider = create_provider(provider_name)
            print(f"  ✅ Using provider: {provider_name} ({getattr(provider, 'model', '?')})")
            return provider, provider_name
        except Exception:
            continue
    return None, None


def try_next_provider(current: str):
    """Get the next provider in fallback order after current."""
    from cdr.llm.factory import create_provider

    idx = PROVIDER_ORDER.index(current) if current in PROVIDER_ORDER else -1
    for provider_name in PROVIDER_ORDER[idx + 1 :]:
        try:
            provider = create_provider(provider_name)
            return provider, provider_name
        except Exception:
            continue
    return None, None


def sanitize_report(report_dict: dict) -> dict:
    """Remove any potential sensitive data from report before saving."""
    # Remove any field that might contain API keys or tokens
    sensitive_patterns = ["api_key", "token", "secret", "credential", "authorization"]

    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if not any(pat in k.lower() for pat in sensitive_patterns)
            }
        elif isinstance(obj, list):
            return [_clean(item) for item in obj]
        elif isinstance(obj, str):
            # Scrub anything that looks like an API key
            for prefix in ["sk-", "gsk_", "AIza", "hf_", "csk-", "sk-or-"]:
                if prefix in obj:
                    return "[REDACTED]"
            return obj
        return obj

    return _clean(report_dict)


def validate_schema(report: dict) -> bool:
    """Validate report against JSON Schema."""
    schema_path = ROOT / "schemas" / "report.schema.json"
    try:
        import jsonschema

        schema = json.loads(schema_path.read_text())
        jsonschema.validate(report, schema)
        return True
    except ImportError:
        print("    ⚠️  jsonschema not installed — skipping validation")
        return True
    except Exception as e:
        print(f"    ⚠️  Schema validation: {e}")
        return False


async def run_single_question(
    question: str,
    run_id: str,
    provider,
    provider_name: str,
    output_dir: Path,
) -> dict | None:
    """Run a single pipeline execution and save results."""
    from cdr.orchestration.graph import CDRRunner

    runner = CDRRunner(
        llm_provider=provider,
        model=getattr(provider, "model", "default"),
        output_dir=str(output_dir),
        dod_level=1,  # Exploratory — fastest for demo
    )

    t_start = time.perf_counter()
    try:
        result = await runner.run(
            research_question=question,
            max_results=MAX_RESULTS,
            formats=["json"],
            run_id=run_id,
        )
    except Exception as e:
        print(f"    ❌ Pipeline error: {e}")
        return None
    t_elapsed = time.perf_counter() - t_start

    # The CDRRunner writes cdr_report_{run_id}.json to output_dir.
    # Read that canonical output if it exists.
    canonical_path = output_dir / f"cdr_report_{run_id}.json"
    if canonical_path.exists():
        state_dict = json.loads(canonical_path.read_text())
    elif hasattr(result, "model_dump"):
        state_dict = result.model_dump(mode="json")
    elif isinstance(result, dict):
        state_dict = result
    else:
        state_dict = {"error": "unexpected result type"}

    # Build report from state
    report = {
        "run_id": run_id,
        "question": question,
        "status": state_dict.get("status", "unknown"),
        "status_reason": state_dict.get("status_reason", ""),
        "disclaimer": (
            "⚠️ This report is machine-generated by CDR (Clinical Deep Research). "
            "It is NOT medical advice and should NOT be used for clinical decision-making. "
            "All findings require independent verification by qualified professionals. "
            "See DISCLAIMER.md for full terms."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pico": state_dict.get("pico", {}),
        "prisma_counts": state_dict.get(
            "prisma_counts",
            {
                "records_identified": 0,
                "studies_included": 0,
            },
        ),
        "study_count": state_dict.get("study_count", 0),
        "claim_count": state_dict.get("claim_count", 0),
        "snippet_count": state_dict.get("snippet_count", 0),
        "claims": state_dict.get("claims", []),
        "run_kpis": state_dict.get("run_kpis", {}),
        "composed_hypotheses": state_dict.get("composed_hypotheses", []),
        "answer": state_dict.get("final_answer", ""),
        "metadata": {
            "provider": provider_name,
            "model": getattr(provider, "model", "unknown"),
            "mode": "online",
            "dod_level": 1,
            "latency_seconds": round(t_elapsed, 2),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
    }

    # Sanitize before saving
    report = sanitize_report(report)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sample_report_online.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Validate
    valid = validate_schema(report)

    print(f"    Status:  {report.get('status', '?')}")
    print(f"    Studies: {report.get('study_count', 0)}")
    print(f"    Claims:  {report.get('claim_count', 0)}")
    print(f"    Latency: {t_elapsed:.1f}s")
    print(f"    Schema:  {'✅' if valid else '⚠️'}")
    print(f"    Output:  {out_path.relative_to(ROOT)}")

    return report


async def main() -> int:
    print("🔬  CDR Online Demo — Real Pipeline Execution")
    print("=" * 55)
    print()

    # Find a working provider
    print("  🔑 Finding available LLM provider...")
    provider, provider_name = get_working_provider()
    if provider is None:
        print("  ❌ No LLM provider configured.")
        print("     Set at least one of: GROQ_API_KEY, CEREBRAS_API_KEY,")
        print("     OPENROUTER_API_KEY, GOOGLE_API_KEY, HF_TOKEN")
        return 1
    print()

    results = []
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        run_id = q["id"]
        question = q["question"]
        output_dir = ROOT / "examples" / "output" / "online" / run_id

        # Skip if output already exists (resume support)
        existing = output_dir / f"cdr_report_{run_id}.json"
        if existing.exists():
            print(f"  [{i}/{len(DEMO_QUESTIONS)}] {run_id}: ⏭ Already exists, skipping")
            results.append({"run_id": run_id, "status": "completed (cached)"})
            continue

        print(f"  [{i}/{len(DEMO_QUESTIONS)}] {run_id}: {question[:55]}...")

        report = await run_single_question(question, run_id, provider, provider_name, output_dir)

        if report:
            results.append({"run_id": run_id, "status": report.get("status", "?")})
        else:
            # Try fallback provider
            print(f"    🔄 Trying fallback provider...")
            fb_provider, fb_name = try_next_provider(provider_name)
            if fb_provider:
                print(f"    ✅ Fallback to: {fb_name}")
                report = await run_single_question(
                    question, run_id, fb_provider, fb_name, output_dir
                )
                if report:
                    results.append({"run_id": run_id, "status": report.get("status", "?")})
                    # Switch to this provider for remaining runs
                    provider, provider_name = fb_provider, fb_name
                else:
                    results.append({"run_id": run_id, "status": "error"})
            else:
                results.append({"run_id": run_id, "status": "error"})

        # Small delay between runs to avoid rate limits
        if i < len(DEMO_QUESTIONS):
            print("    ⏳ Cooling down (5s)...")
            await asyncio.sleep(5)
        print()

    # Summary
    print("=" * 55)
    print("📋  Summary:")
    for r in results:
        emoji = "✅" if r["status"] in ("completed", "insufficient_evidence") else "❌"
        print(f"    {emoji} {r['run_id']}: {r['status']}")
    print()

    succeeded = sum(1 for r in results if r["status"] != "error")
    print(f"  {succeeded}/{len(DEMO_QUESTIONS)} runs completed")
    print(f"  Provider: {provider_name}")
    print()

    if succeeded >= 5:
        print("✅  Online demo complete — all 5 runs succeeded")
    elif succeeded > 0:
        print(f"⚠️   {succeeded} runs succeeded ({len(DEMO_QUESTIONS) - succeeded} failed)")
    else:
        print("❌  All runs failed")

    return 0 if succeeded >= 5 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

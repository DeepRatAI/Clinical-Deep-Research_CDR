#!/usr/bin/env python3
"""
CDR Real Export — Run a live pipeline and export via CDR's built-in Publisher.

This script demonstrates the full CDR export chain:
  1. CDRRunner executes the complete 13-node pipeline (real PubMed/CT.gov data)
  2. CDR's built-in Publisher generates Markdown, JSON, and HTML reports

The exported files are produced by CDR's own reporting module
(cdr.publisher.Publisher), NOT by any external script.

Produces:
    examples/output/online/run_export/
        cdr_report_run_export.json   ← pipeline node output (automatic)
        cdr_report_run_export.md     ← Publisher Markdown export
        cdr_report_run_export.html   ← Publisher HTML export
        cdr_report_run_export_publisher.json ← Publisher JSON export

Usage:
    PYTHONPATH=src python scripts/run_real_export.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load environment
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

OUTPUT_DIR = ROOT / "examples" / "output" / "online" / "run_export"
RUN_ID = "run_export"
MAX_RESULTS = 15

# Clinical question
QUESTION = (
    "What is the evidence for cognitive behavioral therapy (CBT) "
    "versus pharmacotherapy in the treatment of major depressive disorder?"
)

# Provider fallback order (same as run_online_demo.py)
PROVIDER_ORDER = ["openrouter", "groq", "cerebras", "gemini", "huggingface"]

# API key env var mapping
KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
}

# Patterns to scrub from exported files (safety net)
SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"csk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
]


def sanitize_text(text: str) -> str:
    """Replace any leaked API key pattern with [REDACTED]."""
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def pick_provider():
    """Find the first available provider with a configured API key."""
    from cdr.llm.factory import create_provider

    for name in PROVIDER_ORDER:
        env_key = KEY_MAP.get(name, "")
        api_key = os.environ.get(env_key, "").strip()
        if not api_key:
            print(f"  ⏭  {name}: no key ({env_key})")
            continue
        try:
            provider = create_provider(name)
            print(f"  ✅ Using provider: {name}")
            return provider, name
        except Exception as exc:
            print(f"  ⚠️  {name}: init failed ({exc})")
    return None, None


async def run_pipeline_and_export():
    """Execute CDR pipeline and then export via Publisher."""
    from cdr.orchestration.graph import CDRRunner

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Pick LLM provider ──
    print("\n🔍 Selecting LLM provider...")
    provider, provider_name = pick_provider()
    if provider is None:
        print("❌ No LLM provider available. Set at least one API key in .env")
        sys.exit(1)

    # ── 2. Run CDR pipeline ──
    print(f"\n🚀 Running CDR pipeline (run_id={RUN_ID})")
    print(f"   Question: {QUESTION[:80]}...")
    print(f"   Provider: {provider_name}")
    print(f"   Max results: {MAX_RESULTS}")
    print()

    runner = CDRRunner(
        llm_provider=provider,
        output_dir=str(OUTPUT_DIR),
        dod_level=1,
    )

    t0 = time.time()
    final_state = await runner.run(
        research_question=QUESTION,
        max_results=MAX_RESULTS,
        formats=["markdown", "json"],  # pipeline node uses these
        run_id=RUN_ID,
    )
    elapsed = time.time() - t0

    print(f"\n✅ Pipeline completed in {elapsed:.1f}s")
    print(f"   Status:   {final_state.status.value}")
    print(f"   Studies:  {len(final_state.study_cards)}")
    print(f"   Claims:   {len(final_state.claims)}")
    print(f"   Snippets: {len(final_state.snippets)}")

    # ── 3. Export via CDR's built-in Publisher ──
    print("\n📄 Exporting via CDR Publisher module...")

    from cdr.publisher import Publisher

    publisher = Publisher(
        output_dir=OUTPUT_DIR,
        include_appendices=True,
        include_verification=True,
    )

    # Build verification dict {claim_id: VerificationResult}
    verification_dict = {}
    if final_state.verification:
        for vr in final_state.verification:
            verification_dict[vr.claim_id] = vr

    # Synthesis result is on state
    synthesis_result = final_state.synthesis_result
    if synthesis_result is None:
        # Fallback: build a minimal SynthesisResult from claims
        from cdr.core.schemas import SynthesisResult

        synthesis_result = SynthesisResult(
            claims=final_state.claims,
            overall_narrative=final_state.answer or "No narrative available.",
        )
        print("   ⚠️  No synthesis_result on state — built from claims")

    # Critique: state.critique is a Critique object (not CritiqueResult)
    # Publisher expects the "CritiqueResult" from skeptic — which is really
    # the same Critique wrapper. Pass None if types don't align; the Publisher
    # handles None gracefully.
    critique_arg = None
    if final_state.critique:
        # The Publisher accesses .overall_confidence, .key_concerns, .strengths,
        # .recommendation — which are on a different type than schemas.Critique.
        # Passing None is safe; Markdown will say "Limitations not formally assessed".
        critique_arg = None
        print("   ℹ️  Critique available but skipped (type mismatch with Publisher)")

    result = publisher.publish(
        state=final_state,
        synthesis_result=synthesis_result,
        critique_result=critique_arg,
        verification_results=verification_dict if verification_dict else None,
        formats=["markdown", "json", "html"],
    )

    # ── 4. Sanitize exported files ──
    print("\n🔒 Sanitizing exported files for secrets...")
    for fmt, path in result.output_files.items():
        if path and path.exists():
            content = path.read_text(encoding="utf-8")
            clean = sanitize_text(content)
            if clean != content:
                path.write_text(clean, encoding="utf-8")
                print(f"   🔒 {fmt}: secrets redacted")
            else:
                print(f"   ✅ {fmt}: clean")

    # ── 5. Summary ──
    print("\n" + "=" * 60)
    print("📦 CDR Real Export — Results")
    print("=" * 60)
    print(f"  Run ID:    {RUN_ID}")
    print(f"  Provider:  {provider_name}")
    print(f"  Latency:   {elapsed:.1f}s")
    print(f"  Status:    {final_state.status.value}")
    print(f"  Studies:   {len(final_state.study_cards)}")
    print(f"  Claims:    {len(final_state.claims)}")
    print(f"  Snippets:  {len(final_state.snippets)}")
    print()
    print("  Exported files (by CDR Publisher):")

    # Pipeline JSON (written by publish_node automatically)
    pipeline_json = OUTPUT_DIR / f"cdr_report_{RUN_ID[:8]}.json"
    if pipeline_json.exists():
        sz = pipeline_json.stat().st_size
        print(f"    📋 {pipeline_json.name:<40} {sz:>8,} bytes  (pipeline node)")

    # Publisher exports
    for fmt in ["json", "markdown", "html"]:
        path = result.output_files.get(fmt)
        if path and path.exists():
            sz = path.stat().st_size
            label = {"json": "Publisher JSON", "markdown": "Markdown", "html": "HTML"}[fmt]
            print(f"    📋 {path.name:<40} {sz:>8,} bytes  ({label})")

    print()
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline_and_export())

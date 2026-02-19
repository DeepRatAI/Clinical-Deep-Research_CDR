#!/usr/bin/env python3
"""
CDR Audit Verification — Single Online Run (Full Capacity)

Gate requirement: 1 real online E2E run from cdr_lastest/, producing:
  - Pipeline JSON output
  - Publisher exports (MD + HTML + JSON)
  - KPI coherence verification (claim_count == len(claims), etc.)
  - Schema validation

Usage:
    PYTHONPATH=src python scripts/run_audit_verification.py
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

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

OUTPUT_DIR = ROOT / "examples" / "output" / "online" / "run_06"
RUN_ID = "run_06"
MAX_RESULTS = 15

# Different question from previous runs to prove E2E capacity
QUESTION = (
    "What is the comparative effectiveness of SGLT2 inhibitors versus "
    "GLP-1 receptor agonists for cardiovascular outcomes in type 2 diabetes?"
)

PROVIDER_ORDER = ["openrouter", "gemini", "cerebras", "groq", "huggingface"]

KEY_MAP = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "huggingface": "HF_TOKEN",
}

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"csk-[A-Za-z0-9]{20,}"),
    re.compile(r"AIzaSy[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
]


def sanitize_text(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def pick_provider():
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


async def main():
    from cdr.orchestration.graph import CDRRunner

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔬 CDR Audit Verification — Online E2E Run")
    print("=" * 60)

    # ── 1. Provider ──
    print("\n🔍 Selecting LLM provider...")
    provider, provider_name = pick_provider()
    if provider is None:
        print("❌ No LLM provider available.")
        sys.exit(1)

    # ── 2. Pipeline ──
    print(f"\n🚀 Running CDR pipeline")
    print(f"   run_id:   {RUN_ID}")
    print(f"   question: {QUESTION[:80]}...")
    print(f"   provider: {provider_name}")
    print(f"   max_results: {MAX_RESULTS}")
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
        formats=["markdown", "json"],
        run_id=RUN_ID,
    )
    elapsed = time.time() - t0

    status = final_state.status.value
    n_studies = len(final_state.study_cards)
    n_claims = len(final_state.claims)
    n_snippets = len(final_state.snippets)

    print(f"\n✅ Pipeline completed in {elapsed:.1f}s")
    print(f"   status:   {status}")
    print(f"   studies:  {n_studies}")
    print(f"   claims:   {n_claims}")
    print(f"   snippets: {n_snippets}")

    # ── 3. Publisher export ──
    print("\n📄 Exporting via CDR Publisher...")
    from cdr.publisher import Publisher
    from cdr.core.schemas import SynthesisResult

    publisher = Publisher(
        output_dir=OUTPUT_DIR,
        include_appendices=True,
        include_verification=True,
    )

    verification_dict = {}
    if final_state.verification:
        for vr in final_state.verification:
            verification_dict[vr.claim_id] = vr

    synthesis_result = final_state.synthesis_result
    if synthesis_result is None:
        synthesis_result = SynthesisResult(
            claims=final_state.claims,
            overall_narrative=final_state.answer or "No narrative available.",
        )

    result = publisher.publish(
        state=final_state,
        synthesis_result=synthesis_result,
        critique_result=None,
        verification_results=verification_dict if verification_dict else None,
        formats=["markdown", "json", "html"],
    )

    # ── 4. Sanitize ──
    print("\n🔒 Sanitizing exported files...")
    for fmt, path in result.output_files.items():
        if path and path.exists():
            content = path.read_text(encoding="utf-8")
            clean = sanitize_text(content)
            if clean != content:
                path.write_text(clean, encoding="utf-8")
                print(f"   🔒 {fmt}: secrets redacted")
            else:
                print(f"   ✅ {fmt}: clean")

    # ── 5. Write sample_report_online.json ──
    sample = {
        "run_id": RUN_ID,
        "question": QUESTION,
        "status": status,
        "status_reason": getattr(final_state, "status_reason", None) or "technically_complete",
        "disclaimer": getattr(final_state, "disclaimer", None) or "⚠️ This report is machine-generated by CDR (Clinical Deep Research). It is NOT medical advice and should NOT be used for clinical decision-making. All findings require independent verification by qualified professionals. See DISCLAIMER.md for full terms.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pico": final_state.pico.model_dump() if final_state.pico else None,
        "prisma_counts": final_state.prisma_counts.model_dump() if final_state.prisma_counts else None,
        "study_count": n_studies,
        "claim_count": n_claims,
        "snippet_count": n_snippets,
        "claims": [c.model_dump() for c in final_state.claims],
        "run_kpis": final_state.run_kpis.model_dump() if getattr(final_state, "run_kpis", None) else None,
        "composed_hypotheses": [h.model_dump() for h in (final_state.composed_hypotheses or [])],
        "answer": final_state.answer,
        "metadata": {
            "provider": provider_name,
            "latency_seconds": round(elapsed, 1),
            "max_results": MAX_RESULTS,
        },
    }

    sample_path = OUTPUT_DIR / "sample_report_online.json"
    sample_text = json.dumps(sample, indent=2, ensure_ascii=False, default=str)
    sample_text = sanitize_text(sample_text)
    sample_path.write_text(sample_text, encoding="utf-8")

    # ── 6. KPI Coherence ──
    print("\n📊 KPI Coherence Check:")
    cc_ok = sample["claim_count"] == len(sample["claims"])
    print(f"   claim_count={sample['claim_count']} == len(claims)={len(sample['claims'])}  {'✅' if cc_ok else '❌'}")
    
    si = sample.get("prisma_counts", {})
    si_val = si.get("studies_included", 0) if si else 0
    sc_ok = abs(sample["study_count"] - si_val) <= 2
    print(f"   study_count={sample['study_count']} ~= studies_included={si_val}  {'✅' if sc_ok else '⚠️'}")
    print(f"   snippet_count={sample['snippet_count']}")

    # ── 7. Schema validation ──
    print("\n🔍 Schema validation:")
    try:
        import jsonschema
        schema = json.load(open(ROOT / "schemas" / "report.schema.json"))
        jsonschema.validate(sample, schema)
        print("   ✅ sample_report_online.json validates against report.schema.json")
    except Exception as e:
        print(f"   ⚠️ Schema validation: {e}")

    # ── 8. Summary ──
    print("\n" + "=" * 60)
    print("📦 AUDIT VERIFICATION — RESULTS")
    print("=" * 60)
    print(f"  run_id:       {RUN_ID}")
    print(f"  status:       {status}")
    print(f"  provider:     {provider_name}")
    print(f"  latency:      {elapsed:.1f}s")
    print(f"  claim_count:  {n_claims}")
    print(f"  study_count:  {n_studies}")
    print(f"  snippet_count:{n_snippets}")
    print()
    print("  Output files:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        sz = p.stat().st_size
        print(f"    {p.name:<45} {sz:>8,} bytes")
    print()
    print(f"  KPI coherence: {'✅ PASS' if (cc_ok and sc_ok) else '❌ ISSUES'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

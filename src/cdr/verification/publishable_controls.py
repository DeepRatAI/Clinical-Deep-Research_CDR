"""
DoD3 Publishable Controls - Real Evidence Queries

This module contains queries anchored to KNOWN RCTs that SHOULD produce
publishable runs when executed against the CDR system with real retrieval.

The purpose is to validate the "happy path" - confirming that DoD3 gates
correctly PASS for legitimate evidence, not just correctly BLOCK invalid evidence.

Queries are designed to:
1. Anchor to specific, well-known RCTs (JUPITER, EMPEROR-Preserved, DPP)
2. Use explicit comparators (placebo, active control)
3. Target populations that exactly match the trial inclusion criteria
4. Request study types that match the actual evidence

If these queries return UNPUBLISHABLE, the problem is likely:
- PICO-match gate too strict (population matching too literal)
- Arms/comparator extraction failing
- Study type detection errors

Refs:
- DoD3 Contract
- CDR_Agent_Guidance_and_Development_Protocol.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PublishableQuery:
    """A query expected to produce a PUBLISHABLE run."""

    query_id: str
    name: str
    research_question: str
    expected_status: Literal["publishable", "partially_publishable"]

    # Known RCT anchors
    anchor_trials: list[str]  # e.g., ["JUPITER", "NCT00239681"]
    anchor_pmids: list[str]  # e.g., ["18997196"]

    # Expected characteristics
    expected_study_types: list[str]  # ["RCT"]
    expected_comparator: str
    expected_population: str

    # Debugging hints if it fails
    if_fails_check: list[str]

    # DoD level
    dod_level: int = 3


# =============================================================================
# ANCHOR QUERIES - Known publishable patterns
# =============================================================================

PUBLISHABLE_QUERIES = [
    # -------------------------------------------------------------------------
    # Query 1: JUPITER Trial - Rosuvastatin in elevated CRP
    # This is a LANDMARK trial with clear placebo comparator
    # -------------------------------------------------------------------------
    PublishableQuery(
        query_id="PUB-001",
        name="JUPITER Trial Pattern",
        research_question=(
            "rosuvastatin 20mg versus placebo for major cardiovascular events "
            "in adults with elevated C-reactive protein and LDL cholesterol below 130 mg/dL"
        ),
        expected_status="publishable",
        anchor_trials=["JUPITER", "NCT00239681"],
        anchor_pmids=["18997196"],
        expected_study_types=["RCT"],
        expected_comparator="placebo",
        expected_population="adults with elevated CRP and normal LDL",
        if_fails_check=[
            "Check if PubMed retrieval includes PMID 18997196",
            "Verify PICO-match is not penalizing 'elevated CRP' vs 'high-sensitivity CRP'",
            "Check comparator extraction - should detect 'placebo' from abstract",
            "Verify study type detection - should detect 'RCT' from publication_type",
        ],
        dod_level=3,
    ),
    # -------------------------------------------------------------------------
    # Query 2: EMPEROR-Preserved - SGLT2i in HFpEF
    # Recent landmark trial, clear placebo comparator
    # -------------------------------------------------------------------------
    PublishableQuery(
        query_id="PUB-002",
        name="EMPEROR-Preserved Pattern",
        research_question=(
            "empagliflozin versus placebo for hospitalization for heart failure "
            "in patients with heart failure with preserved ejection fraction"
        ),
        expected_status="publishable",
        anchor_trials=["EMPEROR-Preserved", "NCT03057951"],
        anchor_pmids=["34449189"],
        expected_study_types=["RCT"],
        expected_comparator="placebo",
        expected_population="heart failure with preserved ejection fraction",
        if_fails_check=[
            "Check if PubMed retrieval includes PMID 34449189",
            "Verify HFpEF population match (should match EF >= 40%)",
            "Check comparator extraction - 'placebo' should be explicit",
            "Verify primary endpoint detection (hospitalization for HF)",
        ],
        dod_level=3,
    ),
    # -------------------------------------------------------------------------
    # Query 3: DPP - Metformin in Prediabetes
    # Classic prevention trial, very well-known
    # -------------------------------------------------------------------------
    PublishableQuery(
        query_id="PUB-003",
        name="DPP Trial Pattern",
        research_question=(
            "metformin versus placebo for prevention of type 2 diabetes "
            "in adults with prediabetes or impaired glucose tolerance"
        ),
        expected_status="publishable",
        anchor_trials=["DPP", "Diabetes Prevention Program"],
        anchor_pmids=["11832527"],
        expected_study_types=["RCT"],
        expected_comparator="placebo",
        expected_population="adults with prediabetes or IGT",
        if_fails_check=[
            "Check if PubMed retrieval includes PMID 11832527",
            "Verify population match allows 'prediabetes' OR 'impaired glucose tolerance'",
            "Check comparator - DPP had placebo arm",
            "Verify diabetes outcome detection",
        ],
        dod_level=3,
    ),
    # -------------------------------------------------------------------------
    # Query 4: Specific Trial Name Query (fallback if semantic fails)
    # By including trial name, we force high-precision retrieval
    # -------------------------------------------------------------------------
    PublishableQuery(
        query_id="PUB-004",
        name="Trial-Name-Anchored Query",
        research_question=(
            "DELIVER trial: dapagliflozin versus placebo for heart failure outcomes "
            "in patients with heart failure and mildly reduced or preserved ejection fraction"
        ),
        expected_status="publishable",
        anchor_trials=["DELIVER", "NCT03619213"],
        anchor_pmids=["36027570"],
        expected_study_types=["RCT"],
        expected_comparator="placebo",
        expected_population="HF with mildly reduced or preserved EF",
        if_fails_check=[
            "Check if including 'DELIVER' in query improves retrieval precision",
            "If still failing, PICO-match may be too strict on EF cutoffs",
            "Check if trial-name queries bypass some screening filters",
        ],
        dod_level=3,
    ),
]


# =============================================================================
# RUNNER
# =============================================================================


async def run_publishable_harness(
    base_url: str = "http://localhost:8000",
    verbose: bool = True,
) -> dict:
    """
    Execute publishable control queries against running CDR instance.

    Returns:
        Results dict with pass/fail for each query
    """
    import asyncio

    import httpx

    results = {
        "total": len(PUBLISHABLE_QUERIES),
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        for query in PUBLISHABLE_QUERIES:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Running {query.query_id}: {query.name}")
                print(f"{'=' * 60}")

            # Start run
            try:
                response = await client.post(
                    f"{base_url}/api/v1/runs",
                    json={
                        "research_question": query.research_question,
                        "dod_level": query.dod_level,
                        "max_results": 15,
                    },
                )

                if response.status_code != 202:
                    print(f"  ERROR: Failed to start run: {response.text}")
                    results["failed"] += 1
                    results["details"].append(
                        {
                            "query_id": query.query_id,
                            "status": "error",
                            "error": f"HTTP {response.status_code}",
                        }
                    )
                    continue

                run_data = response.json()
                run_id = run_data["run_id"]
                print(f"  Run ID: {run_id}")

                # Poll for completion
                for _ in range(60):  # Max 10 minutes
                    await asyncio.sleep(10)
                    status_response = await client.get(f"{base_url}/api/v1/runs/{run_id}")

                    if status_response.status_code != 200:
                        continue

                    status_data = status_response.json()
                    current_status = status_data.get("status", "unknown")

                    if current_status in (
                        "completed",
                        "publishable",
                        "unpublishable",
                        "insufficient_evidence",
                        "failed",
                    ):
                        break

                    if verbose:
                        print(f"  Status: {current_status}...")

                # Check result
                final_status = status_data.get("status", "unknown")

                if final_status in ("completed", "publishable"):
                    print(f"  ✅ PASS: {final_status}")
                    results["passed"] += 1
                    results["details"].append(
                        {
                            "query_id": query.query_id,
                            "status": "pass",
                            "run_status": final_status,
                            "run_id": run_id,
                        }
                    )
                else:
                    print(f"  ❌ FAIL: {final_status}")
                    print("  Debug hints:")
                    for hint in query.if_fails_check:
                        print(f"    - {hint}")

                    results["failed"] += 1
                    results["details"].append(
                        {
                            "query_id": query.query_id,
                            "status": "fail",
                            "run_status": final_status,
                            "run_id": run_id,
                            "expected": query.expected_status,
                            "debug_hints": query.if_fails_check,
                        }
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                results["failed"] += 1
                results["details"].append(
                    {
                        "query_id": query.query_id,
                        "status": "error",
                        "error": str(e),
                    }
                )

    # Summary
    print(f"\n{'=' * 60}")
    print("PUBLISHABLE HARNESS RESULTS")
    print(f"{'=' * 60}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    if results["failed"] > 0:
        print(f"\n⚠️ {results['failed']} publishable queries failed!")
        print("This indicates DoD3 gates may be too strict or retrieval/extraction issues.")
    else:
        print("\n✅ All publishable queries passed - DoD3 happy path validated!")

    return results


def get_publishable_queries() -> list[PublishableQuery]:
    """Get all publishable query definitions."""
    return PUBLISHABLE_QUERIES


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_publishable_harness())

# ADR-005: Post-Change Audit Corrections

## Status: Accepted (Revised 2026-01-20)

## Date: 2025-01-27 (Revised: 2026-01-20)

## Revision Note

**2026-01-20**: ADR corregido para alinearse con el contrato real de `EvidenceClaim`.
Eliminados campos inexistentes (`statement`, `direction`, `strength`, `verified`, `claim.snippets`, `rob2_domain_ratings`).
Documentación ahora refleja el schema implementado en `src/cdr/core/schemas.py`.

## Context

Following ADR-004 implementation, a post-change audit (`CDR_Post_ADR003_v3_PostChange_Audit_and_Actions.md`) identified 5 residual findings:

| Finding | Severity | Issue |
|---------|----------|-------|
| F1 | CRÍTICO | Published report drops structured evidence rationale |
| F2 | ALTO | `dod_level` only enforced in screening, not end-to-end |
| F3 | ALTO | 18 retrieval tests skipped due to API drift |
| F4 | MEDIO | Abstract length cutoff without full-text fallback |
| F5 | MEDIO | `grade_rationale` no completeness gating |

## Decision

### Action 1: Full Traceability in Published Report

**Location**: `src/cdr/orchestration/graph.py` - `publish_node()`

#### EvidenceClaim Contract (Real Schema)

```python
# src/cdr/core/schemas.py - EvidenceClaim
class EvidenceClaim(BaseModel):
    claim_id: str
    claim_text: str  # min 20, max 1000 chars
    certainty: GRADECertainty
    
    # Evidence support (REQUIRED - min 1)
    supporting_snippet_ids: list[str]
    
    # Conflicts
    conflicting_snippet_ids: list[str] = []
    
    # Qualifications
    limitations: list[str] = []
    
    # GRADE Rationale (structured)
    # Keys: risk_of_bias, inconsistency, indirectness, imprecision, publication_bias
    grade_rationale: dict[str, str] = {}
    
    # Study counts
    studies_supporting: int = 0
    studies_conflicting: int = 0
```

#### Published Report Structure (report_data)

```python
report_data = {
    "run_id": str,
    "question": str,
    "pico": {...} | None,
    "prisma_counts": {...} | None,
    "study_count": int,
    "claim_count": int,
    "snippet_count": int,
    
    # FULL CLAIM TRACEABILITY - aligned with EvidenceClaim schema
    "claims": [
        {
            "claim_id": str,
            "claim_text": str,
            "certainty": str,  # GRADECertainty.value
            "supporting_snippet_ids": list[str],
            "conflicting_snippet_ids": list[str],
            "limitations": list[str],
            "grade_rationale": dict[str, str],
            "studies_supporting": int,
            "studies_conflicting": int,
            # Computed linkage
            "linked_records": list[str],  # record_ids extracted from snippet_ids
            "linked_rob2": [
                {
                    "record_id": str,
                    "overall_judgment": str | None,
                    "overall_rationale": str | None,
                }
            ],
        }
    ],
    
    # RoB2 summary (all studies)
    "rob2_summary": [
        {
            "record_id": str,
            "overall_judgment": str | None,
            "overall_rationale": str | None,
            "domains": [
                {"domain": str, "judgment": str, "rationale": str}
            ],
        }
    ],
    
    # Verification results
    "verification_summary": [
        {
            "claim_id": str,
            "overall_status": str,
            "overall_confidence": float,
            "checks_count": int,
        }
    ],
    
    # Run KPIs
    "run_kpis": {
        "snippet_coverage": float,
        "verification_coverage": float,
        "claims_with_evidence_ratio": float,
        "total_claims": int,
        "verified_claims": int,
        "unverified_claims": int,
        "is_negative_outcome": bool,
        "dod_level": int,
    },
    
    "critique_findings": int,
    "critique_blockers": list,
    "errors": list,
    "status": str,
    "status_reason": str,
}
```

### Action 2: DoD End-to-End Enforcement

**Locations Modified**:
1. `CDRRunner.__init__()` - accepts `dod_level: int = 1`
2. `CDRRunner.run()` - accepts optional `dod_level` override
3. `synthesize_node()` - reads `dod_level` from configurable, **applies EARLY gates**
4. `publish_node()` - enforces FINAL DoD gates

#### DoD Level Gates

| Level | Node | Gate | Behavior |
|-------|------|------|----------|
| 1 (exploratory) | synthesize | None | Markdown fallback allowed |
| 2 (research-grade) | synthesize | JSON required | Markdown fallback → INSUFFICIENT_EVIDENCE |
| 2 (research-grade) | publish | verification_coverage >= 80% | Partial status if violated |
| 3 (SOTA-grade) | synthesize | grade_rationale required | Claims without rationale blocked |
| 3 (SOTA-grade) | publish | All Level 2 + rationale | Final validation |

#### Early Gate Implementation (synthesize_node)

```python
dod_level = configurable.get("dod_level", 1)

# Level 2+ gate: JSON synthesis required, no Markdown fallback
if dod_level >= 2 and used_markdown_fallback:
    # Block synthesis - cannot produce claims with heuristic parsing
    return {
        "claims": [],
        "response": None,
        "errors": ["DOD_LEVEL_2_JSON_REQUIRED: Markdown fallback not allowed for Research-grade"],
        "synthesis_metadata": {
            "blocked_by_dod": True,
            "dod_level": dod_level,
            "reason_code": "MARKDOWN_FALLBACK_NOT_ALLOWED",
        },
    }

# Level 3 gate: grade_rationale required for ALL claims
if dod_level >= 3:
    claims_missing_rationale = [c for c in claims if not c.grade_rationale]
    if claims_missing_rationale:
        return {
            "claims": [],
            "response": None,
            "errors": [
                f"DOD_LEVEL_3_GRADE_RATIONALE_REQUIRED: {len(claims_missing_rationale)} claims missing rationale"
            ],
            "synthesis_metadata": {
                "blocked_by_dod": True,
                "dod_level": dod_level,
                "reason_code": "GRADE_RATIONALE_MISSING",
                "claims_blocked": [c.claim_id for c in claims_missing_rationale],
            },
        }
```

#### Final Gate Implementation (publish_node)

```python
dod_level = configurable.get("dod_level", 1)

# Level 2 gate: verification coverage
if dod_level >= 2 and verification_coverage < 0.8:
    final_status = RunStatus.UNPUBLISHABLE
    status_reason = f"DoD Level 2 requires >=80% verification; got {verification_coverage*100:.1f}%"
    state["warnings"].append(status_reason)

# Level 3 gate: final grade_rationale check (defense in depth)
if dod_level >= 3:
    claims_missing_rationale = [c for c in state.claims if not c.grade_rationale]
    if claims_missing_rationale:
        final_status = RunStatus.UNPUBLISHABLE
        status_reason = f"DoD Level 3: {len(claims_missing_rationale)} claims missing grade_rationale"
        state["warnings"].append(status_reason)
```

### Action 3: Retrieval Tests Rewrite

**Problem**: 18 tests in `tests/test_retrieval.py` were skipped due to:
- Internal API methods renamed/removed
- API signature changes (QdrantStore, Reranker)
- Missing pytest-asyncio configuration

**Solution**: Complete rewrite using public API + deterministic mocks.

**Changes**:

| Component | Original | New |
|-----------|----------|-----|
| PubMedClient | 4 skipped | 6 tests (init, params, filters) |
| ClinicalTrialsClient | 3 skipped | 6 tests (init, filters, combined) |
| Embedder | 3 skipped | 6 tests (init, embed, batch, missing lib) |
| Reranker | 3 skipped | 8 tests (init, score, batch, rerank API) |
| QdrantStore | 3 skipped | 8 tests (init, collection, upsert) |
| Integration | 2 skipped | 3 tests (pipeline, hybrid, cache) |

**Test Suite Results**:
- Before ADR-005: 99 passed, 18 skipped (117 total)
- After ADR-005 (initial): 136 passed, 0 skipped
- After ADR-005 (revision 2026-01-20): 145 passed, 0 skipped

### Action 4 & 5: Abstract Cutoff and Grade Rationale Gating

**F4 (Abstract cutoff)**: Deferred to future work. Current implementation uses full abstract when available; full-text fallback requires PDF parsing infrastructure.

**F5 (Grade rationale gating)**: Implemented as DoD Level 3 **early gate** in `synthesize_node()` (blocks before publish).

## Changes (Revision 2026-01-20)

### Schema Changes

1. **`SynthesisResult.used_markdown_fallback: bool`** (new field)
   - Location: `src/cdr/core/schemas.py`
   - Purpose: Track if claims were extracted via heuristic Markdown parsing
   - Used by: DoD Level 2+ gate in `synthesize_node()`

2. **`synthesizer._parse_synthesis_response()`** updated
   - Location: `src/cdr/synthesis/synthesizer.py`
   - Changes: Sets `used_markdown_fallback=True` for fallback path, `False` for JSON path
   - Edge case: JSON parsed but 0 claims → NOT fallback (valid empty result)

### Early Gate Implementation

Gates now applied in `synthesize_node()` BEFORE claims reach `publish_node()`:

1. **Level 2+ Gate**: Blocks if `result.used_markdown_fallback is True`
2. **Level 3 Gate**: Blocks if any claim has empty `grade_rationale`

Both gates emit:
- Error message with reason code (`DOD_LEVEL_2_JSON_REQUIRED`, `DOD_LEVEL_3_GRADE_RATIONALE_REQUIRED`)
- Span attributes for observability (`dod_blocked`, `dod_block_reason`)
- Empty claims list to prevent downstream processing

## Consequences

### Positive
- Published reports now include complete audit trail for regulatory review
- DoD levels enforce quality gates throughout the pipeline
- **Early gates prevent wasted compute on invalid syntheses**
- Test suite has zero skipped tests, improving CI/CD reliability
- KPIs enable automated quality monitoring

### Negative
- DoD Level 3 may reject valid reports if LLM fails to generate rationale
- Report JSON size increased ~40% due to additional fields
- Early blocking may require retry with different LLM parameters

### Risks Mitigated
- **Traceability gap**: Regulators can now trace each claim to evidence and RoB2 ratings
- **Quality drift**: DoD gates prevent publishing substandard reports
- **Test rot**: Public API tests are less fragile than internal API tests
- **Late failure**: Early gates catch issues before expensive verification

## Verification

```bash
# Full test suite (145 tests)
python -m pytest tests/ --tb=short -q
# Result: 145 passed, 0 skipped

# DoD early gate tests specifically (9 tests)
python -m pytest tests/test_synthesis.py -v -k "DoD"
# Result: 9 passed

# Retrieval tests (41 tests)
python -m pytest tests/test_retrieval.py -v
# Result: 41 passed
```

## Related ADRs
- ADR-002: SOTA-Grade Corrections
- ADR-003: Audit Corrections v2
- ADR-004: Audit Corrections v3 (predecessor)

## References
- PRISMA 2020 Statement (Page et al., 2021)
- GRADE Handbook (Schünemann et al., 2013)
- Cochrane RoB2 Tool (Sterne et al., 2019)

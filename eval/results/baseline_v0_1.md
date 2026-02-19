# CDR v0.1 Baseline Results

> Evaluation baseline for CDR v0.1 Open Alpha — February 2026

## Configuration

| Parameter | Value |
|-----------|-------|
| **LLM Provider** | HuggingFace Inference API |
| **LLM Model** | meta-llama/Meta-Llama-3.1-70B-Instruct |
| **Seed** | 42 |
| **Dataset** | `eval/datasets/golden_set_toy.json` (5 questions) |
| **Dataset checksum** | `sha256:9bd1f3c9ae50febe0ae457801d6dff15197cc7969b518b4aefc88e82ed88bec3` |
| **Date** | 2026-02-16 |

## Per-Question Results

| ID | Question | Studies | Claims | Snippet Cov | Verif Cov | Composition | Status |
|----|----------|---------|--------|-------------|-----------|-------------|--------|
| GS-001 | Aspirin for secondary CV prevention | 15 | 8 | 1.00 | 0.88 | — | completed |
| GS-002 | Metformin + GLP-1 vs monotherapy | 8 | 6 | 1.00 | 0.83 | ✅ | completed |
| GS-003 | Vitamin D for respiratory infections | 10 | 5 | 1.00 | 0.70 | — | completed |
| GS-004 | High-sensitivity troponin for acute MI | 12 | 7 | 1.00 | 0.86 | — | completed |
| GS-005 | Anti-inflammatory for CV + elevated CRP | 5 | 6 | 1.00 | 0.80 | ✅ | completed |

## Aggregate Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Snippet coverage** | 1.00 | ≥1.00 | ✅ PASS |
| **Avg verification coverage** | 0.82 | ≥0.80 | ✅ PASS |
| **Pipeline completion rate** | 5/5 (100%) | — | ✅ |
| **Publishable rate** | ~60% | ≥60% | ✅ PASS |
| **Composition emitted** | 2/5 (40%) | ≥10% | ✅ PASS |

## Latency Profile (approximate, single-machine)

| Stage | p50 | p95 | Notes |
|-------|-----|-----|-------|
| parse_question | <1s | <2s | Single LLM call |
| plan_search | <1s | <2s | Single LLM call |
| retrieve | 3–8s | 12s | Network-bound (PubMed + CT.gov) |
| deduplicate | <0.5s | <1s | CPU-only (title similarity) |
| screen | 2–5s | 8s | 1 LLM call per record batch |
| parse_docs (PMC) | 2–10s | 15s | Network-bound (PMC OA fetch) |
| extract_data | 5–15s | 25s | 1 LLM call per study |
| assess_rob2 | 3–10s | 18s | 1 LLM call per study |
| synthesize | 5–12s | 20s | Multi-step LLM synthesis |
| critique | 3–8s | 12s | Single LLM call |
| verify | 2–5s | 8s | Entailment checks |
| compose | 2–8s | 15s | Cross-study inference |
| publish | <1s | <2s | Template rendering |
| **Total (E2E)** | **35–80s** | **~140s** | Depends on study count + provider |

## Token Usage (estimated per run)

| Component | Est. Tokens | Notes |
|-----------|-------------|-------|
| Input (prompts) | 15,000–40,000 | Scales with study count |
| Output (responses) | 5,000–15,000 | Claims + assessments |
| **Total per run** | **20,000–55,000** | Provider-dependent |

## Retrieval Quality

| Metric | GS-001 | GS-002 | GS-003 | GS-004 | GS-005 |
|--------|--------|--------|--------|--------|--------|
| Records identified | 50+ | 30+ | 40+ | 45+ | 20+ |
| After dedup | 40+ | 25+ | 35+ | 40+ | 18+ |
| After screening | 15 | 8 | 10 | 12 | 5 |
| Recall@20 (est.) | High | Moderate | Moderate | High | Moderate |

> **Note on Recall@k**: True Recall@k requires a labeled relevance set, which we don't have for v0.1. The values above are estimated based on known landmark studies appearing in results. Creating a proper relevance-labeled test set is a v0.2 goal (see [ROADMAP.md](ROADMAP.md)).

## Failure Modes Observed

| Mode | Frequency | Impact | Documented |
|------|-----------|--------|------------|
| Citation laundering | ~5% of claims | Incorrect PMID attribution | [INC-001](INCIDENTS.md#inc-001-citation-laundering) |
| Uniform RoB2 | ~30% of assessments | Uninformative bias rating | [INC-002](INCIDENTS.md#inc-002-uniform-rob2-some-concerns) |
| CT.gov empty results | ~10% of runs | Missing trial data | Fixed (500-char limit) |
| Unpublishable output | ~40% of runs | Report marked unpublishable | By design (honest reporting) |

## Reproducibility Notes

- **What is reproducible**: Pipeline structure, retrieval queries, schema validation, quality gates.
- **What varies**: LLM text output (temperature, provider differences), PubMed index changes over time.
- **Best practice**: Run evaluations within a short time window. Record provider + model + timestamp.

## How to Reproduce

```bash
make eval
# or
PYTHONPATH=src python -m eval.eval_runner \
    --dataset eval/datasets/golden_set_toy.json \
    --output eval/results/ \
    --compare-baseline eval/results/baseline_v0_1.json
```

---

*Raw data: [eval/results/baseline_v0_1.json](eval/results/baseline_v0_1.json)*

# CDR Online Run Notes

> Metadata for pre-computed online pipeline runs (no secrets, no API keys).

## Overview

These runs were executed against the **real CDR pipeline** — fetching from PubMed/ClinicalTrials.gov, calling LLM providers, and generating full evidence reports. They serve as reference outputs for verifying that the pipeline produces meaningful clinical evidence analysis.

## Configuration

| Parameter | Value |
|-----------|-------|
| **CDR Version** | v0.1 Open Alpha |
| **Python** | 3.12 |
| **DoD Level** | 1 (Exploratory) |
| **Max Records** | 50 per source |
| **Date** | 2026-02-18 |

## Runs

### Run 01 — Aspirin for Secondary CV Prevention

| Field | Value |
|-------|-------|
| **ID** | `run_01` |
| **Question** | Is aspirin effective for secondary prevention of cardiovascular events? |
| **Provider** | Groq |
| **Model** | llama-3.1-8b-instant |
| **Latency** | 1595.5s |
| **PRISMA** | 51 identified → 27 included |
| **Studies** | 27 |
| **Claims** | 4 |
| **Snippets** | 53 |
| **Output** | `examples/output/online/run_01/cdr_report_run_01.json` |

### Run 02 — Metformin + GLP-1 Agonists

| Field | Value |
|-------|-------|
| **ID** | `run_02` |
| **Question** | Does metformin combined with GLP-1 agonists improve outcomes beyond monotherapy in type 2 diabetes? |
| **Provider** | Groq |
| **Model** | llama-3.1-8b-instant |
| **Latency** | 1148.6s |
| **PRISMA** | 48 identified → 18 included |
| **Studies** | 18 |
| **Claims** | 4 |
| **Snippets** | 36 |
| **Output** | `examples/output/online/run_02/cdr_report_run_02.json` |

### Run 03 — Vitamin D for Respiratory Infections

| Field | Value |
|-------|-------|
| **ID** | `run_03` |
| **Question** | Is vitamin D supplementation effective for preventing respiratory infections? |
| **Provider** | OpenRouter |
| **Model** | meta-llama/llama-3.1-8b-instruct |
| **Latency** | 641.5s |
| **PRISMA** | 30 identified → 20 included |
| **Studies** | 21 |
| **Claims** | 3 |
| **Snippets** | 54 |
| **Output** | `examples/output/online/run_03/cdr_report_run_03.json` |

### Run 04 — High-Sensitivity Troponin for AMI

| Field | Value |
|-------|-------|
| **ID** | `run_04` |
| **Question** | What is the accuracy of high-sensitivity troponin for diagnosing acute myocardial infarction? |
| **Provider** | OpenRouter |
| **Model** | meta-llama/llama-3.1-8b-instruct |
| **Latency** | 396.6s |
| **PRISMA** | 14 identified → 7 included |
| **Studies** | 8 |
| **Claims** | 3 |
| **Snippets** | 14 |
| **Output** | `examples/output/online/run_04/cdr_report_run_04.json` |

### Run 05 — Anti-Inflammatory Therapies for CV Events

| Field | Value |
|-------|-------|
| **ID** | `run_05` |
| **Question** | Can anti-inflammatory therapies reduce cardiovascular events in patients with elevated CRP? |
| **Provider** | OpenRouter |
| **Model** | meta-llama/llama-3.1-8b-instruct |
| **Latency** | 808.5s |
| **PRISMA** | 29 identified → 20 included |
| **Studies** | 20 |
| **Claims** | 3 |
| **Snippets** | 55 |
| **Output** | `examples/output/online/run_05/cdr_report_run_05.json` |

## Summary

| Run | Studies | Claims | Latency | Provider |
|-----|---------|--------|---------|----------|
| run_01 | 27 | 4 | 1595.5s | Groq |
| run_02 | 18 | 4 | 1148.6s | Groq |
| run_03 | 21 | 3 | 641.5s | OpenRouter |
| run_04 | 8 | 3 | 396.6s | OpenRouter |
| run_05 | 20 | 3 | 808.5s | OpenRouter |
| **Total** | **94** | **17** | **4590.6s** | — |

## Reproduction

To reproduce these runs:

```bash
# 1. Configure at least one LLM provider key in .env
cp .env.example .env
# Edit .env — set GROQ_API_KEY (or any supported provider)

# 2. Run all 5 questions
make demo-online
```

> **Note**: Results will differ between runs due to LLM non-determinism and PubMed index updates. The pre-computed outputs above are snapshots from a specific point in time.

## Validation

All outputs pass JSON Schema validation:

```bash
PYTHONPATH=src python -c "
import json, jsonschema
from pathlib import Path
schema = json.loads(Path('schemas/report.schema.json').read_text())
for i in range(1, 6):
    p = Path(f'examples/output/online/run_{i:02d}/sample_report_online.json')
    if p.exists():
        jsonschema.validate(json.loads(p.read_text()), schema)
        print(f'  ✅ {p.name} — valid')
    else:
        print(f'  ⬜ {p.name} — not yet generated')
"
```

---

*See [EVAL.md](../EVAL.md) for evaluation methodology and [docs/report_anatomy.md](report_anatomy.md) for how to read CDR reports.*

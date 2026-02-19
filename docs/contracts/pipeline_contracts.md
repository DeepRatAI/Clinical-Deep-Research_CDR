# CDR Stage Contracts

> Input/output specifications for each pipeline node.
> These contracts define what each stage receives and produces.

## Overview

CDR's 13-node pipeline has typed contracts at every boundary. Data flows through Pydantic models — if a contract is violated, the pipeline fails loudly rather than silently producing bad output.

```
parse_question → plan_search → retrieve → deduplicate → screen →
parse_docs → extract_data → assess_rob2 → synthesize →
critique → verify → compose → publish
```

---

## 1. parse_question

**Purpose**: Decompose a natural-language clinical question into structured PICO components.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.question` (str) | Raw clinical question |
| **Output** | `CDRState.pico` (PICO) | Structured PICO with population, intervention, comparator, outcome |

**Invariants**:
- `pico.population`, `pico.intervention`, `pico.outcome` must each be ≥3 chars
- `pico.comparator_source` must be set (user_specified, assumed, inferred, heuristic, or n/a)
- No LLM hallucinated fields — only extract what's in the question

---

## 2. plan_search

**Purpose**: Generate database-specific search queries from PICO.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.pico` (PICO) | Structured clinical question |
| **Output** | `CDRState.search_plan` (SearchPlan) | PubMed query, CT.gov query, date range, limits |

**Invariants**:
- `search_plan.pubmed_query` uses MeSH terms + Boolean operators
- `search_plan.ct_gov_query` ≤500 chars (API limit)
- `search_plan.max_results_per_source` ∈ [1, 1000]
- `search_plan.created_at` is set

---

## 3. retrieve

**Purpose**: Execute searches against PubMed and ClinicalTrials.gov.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.search_plan` (SearchPlan) | Planned queries |
| **Output** | `CDRState.records` (list[Record]) | Retrieved records with metadata |
| **Output** | `CDRState.executed_searches` (list[ExecutedSearch]) | PRISMA-S audit trail |

**Invariants**:
- Each record has unique `record_id`
- Each record has `source` (pubmed, clinical_trials, pmc)
- `executed_searches` contains one entry per API call actually made
- `results_count` and `results_fetched` are accurate

---

## 4. deduplicate

**Purpose**: Remove duplicate records across sources.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.records` (list[Record]) | All retrieved records |
| **Output** | `CDRState.records` (list[Record]) | Deduplicated records |
| **Output** | `CDRState.prisma_counts.duplicates_removed` (int) | Count of duplicates |

**Invariants**:
- Dedup uses PMID matching + title similarity (threshold ≥0.85)
- `len(output_records) + duplicates_removed == len(input_records)`
- Original records are not modified (new list created)

---

## 5. screen

**Purpose**: Include/exclude records based on PICO relevance.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.records` (list[Record]), `CDRState.pico` (PICO) | Records + criteria |
| **Output** | `CDRState.screening_decisions` (list[ScreeningDecision]) | Per-record decisions |

**Invariants**:
- Every record gets exactly one `ScreeningDecision`
- Every exclusion has a non-empty `reason` code (ExclusionReason enum)
- `decision.record_id` matches an existing record
- Records themselves are NOT modified

---

## 6. parse_docs

**Purpose**: Retrieve and parse full text from PMC Open Access.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.records` (list[Record]) | Included records |
| **Output** | `CDRState.snippets` (list[Snippet]) | Extracted text passages |

**Invariants**:
- Each snippet has a `source_ref` linking to a specific record and section
- Snippet `section` uses the Section enum (abstract, methods, results, discussion, etc.)
- No snippet without a traceable source

---

## 7. extract_data

**Purpose**: Extract structured study data from text.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.snippets` (list[Snippet]), `CDRState.records` (list[Record]) | Evidence text |
| **Output** | `CDRState.study_cards` (list[StudyCard]) | Structured study summaries |

**Invariants**:
- Each study card links to a specific record via `record_id`
- Study type (RCT, cohort, etc.) is determined and recorded
- Sample size, outcomes, and effect sizes are extracted when available

---

## 8. assess_rob2

**Purpose**: Assess risk of bias using RoB2 (RCTs) or ROBINS-I (observational).

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.study_cards`, `CDRState.snippets` | Study data + text |
| **Output** | `CDRState.rob2_results` (list[RoB2Result]) | Bias assessments for RCTs |
| **Output** | `CDRState.robins_i_results` (list[ROBINSIResult]) | Bias assessments for non-RCTs |

**Invariants**:
- RoB2: exactly 5 domains, each with judgment + rationale
- ROBINS-I: exactly 7 domains, each with judgment + rationale
- Overall judgment is set and consistent with domain judgments
- When only abstract is available, rationale states this limitation

---

## 9. synthesize

**Purpose**: Generate evidence claims from extracted data.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.study_cards`, `CDRState.snippets`, `CDRState.rob2_results` | All evidence |
| **Output** | `CDRState.claims` (list[EvidenceClaim]) | Evidence claims with traceability |
| **Output** | `CDRState.synthesis` (SynthesisResult) | Synthesis metadata |

**Invariants**:
- Every claim has `supporting_snippet_ids` (≥1 snippet)
- Claim text is 20-1000 chars
- `certainty` uses GRADE vocabulary (high, moderate, low, very_low)
- `grade_rationale` populated for each claim

---

## 10. critique

**Purpose**: Adversarial review of synthesized claims.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.claims`, `CDRState.snippets` | Claims + evidence |
| **Output** | `CDRState.critiques` (list[Critique]) | Critique findings |

**Invariants**:
- Each critique has `dimension` (from CritiqueDimension enum)
- Each critique has `severity` (blocker, major, minor, observation)
- Blockers must be addressed before publication

---

## 11. verify

**Purpose**: Check citation coverage and claim-source entailment.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.claims`, `CDRState.snippets` | Claims + evidence |
| **Output** | `CDRState.verification_results` (dict[str, VerificationResult]) | Per-claim verification |

**Invariants**:
- Every claim gets a verification result
- `overall_status` ∈ {verified, partial, contradicted, unverifiable}
- `overall_confidence` ∈ [0.0, 1.0]
- Citation coverage = verified_claims / total_claims

---

## 12. compose

**Purpose**: Generate compositional hypotheses (A+B⇒C) from cross-study evidence.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState.claims` | Verified claims |
| **Output** | `CDRState.composed_hypotheses` (list[ComposedHypothesis]) | Novel hypotheses |

**Invariants**:
- Each hypothesis links to ≥2 source claims
- `confidence_score` ∈ [0.0, 1.0]
- `strength` ∈ {strong, moderate, weak}
- `reasoning_trace` explains the inferential chain
- Hypotheses are clearly marked as **inferences**, not established facts

---

## 13. publish

**Purpose**: Assemble final report with PRISMA flowchart, quality gates, and output format.

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `CDRState` (complete) | All pipeline artifacts |
| **Output** | `CDRState.report` (dict) | Final JSON report conforming to `schemas/report.schema.json` |
| **Output** | `CDRState.status` (RunStatus) | Final status |

**Invariants**:
- Report JSON validates against `schemas/report.schema.json`
- PRISMA counts are computed atomically from final state
- Clinical disclaimer is included
- Status reflects actual quality: `unpublishable` if gates fail
- `status_reason` explains the status in machine-readable terms

---

*For the JSON Schema of the final report, see [schemas/report.schema.json](../schemas/report.schema.json).*

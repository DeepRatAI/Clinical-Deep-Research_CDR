# CDR Case Study: Building an Evidence Engine for Clinical Research

> Technical narrative for v0.1 Open Alpha — February 2026

## Problem / Context

Systematic reviews are the gold standard of clinical evidence, but they take **months of manual work**: searching databases, screening thousands of abstracts, extracting data, assessing bias, and synthesizing findings. Most published reviews are already outdated by the time they appear.

CDR explores whether **a well-structured LLM pipeline can automate the mechanical parts** of systematic review — retrieval, screening, extraction, bias assessment — while preserving the traceability and rigor that makes reviews trustworthy.

**The core bet**: it's possible to build an automated system where every claim in the output is traceable to a specific passage in a specific study, and where the system honestly reports its limitations rather than hallucinating confidence.

## Constraints & Non-Goals

**What CDR is:**
- A research exploration tool for scientists
- A pipeline that automates the tedious parts of evidence gathering
- An honest system that reports uncertainty and limitations

**What CDR is NOT:**
- A medical device or clinical decision-support system
- A replacement for expert judgment or peer review
- A system that should be trusted without verification
- A complete PRISMA-compliant systematic review tool (yet)

**Design constraints:**
- Must work with free/public data sources (PubMed, ClinicalTrials.gov, PMC OA)
- Must be reproducible: same question → comparable results
- Must be auditable: every decision must have a trace
- Must be honest: no overclaiming, visible limitations

## Architecture & Boundaries

CDR is a **13-node LangGraph StateGraph** pipeline:

```
parse_question → plan_search → retrieve → deduplicate → screen →
parse_docs → extract_data → assess_rob2 → synthesize →
critique → verify → compose → publish
```

**Boundary decisions:**
- **Retrieval boundary**: PubMed + ClinicalTrials.gov only. We don't cover Embase, Cochrane, or grey literature. This is a known limitation documented in every output.
- **LLM boundary**: CDR is LLM-agnostic (8 providers). The pipeline structure enforces contracts regardless of which LLM is used.
- **Output boundary**: Structured JSON + Markdown + HTML. No interactive exploration (that's a v0.2 goal).
- **Trust boundary**: CDR never claims its output is clinically actionable. Every report carries a disclaimer.

## Key Decisions & Tradeoffs

### 1. LangGraph over bare agents
**Decision**: Use LangGraph StateGraph with explicit nodes and typed state.
**Why**: Predictable execution order, typed interfaces at every step, easy to test individual nodes. Agent-based approaches trade debuggability for flexibility — we chose debuggability.
**Tradeoff**: Less "autonomous" — CDR follows a fixed pipeline, not an agent loop.

### 2. Pydantic contracts over free-form dicts
**Decision**: 20 Pydantic models with strict validation at every node boundary.
**Why**: A dict-based pipeline would be faster to prototype but impossible to maintain. When Node A's output changes, Pydantic immediately tells you what breaks downstream.
**Tradeoff**: More boilerplate, slower iteration early on, but saved dozens of integration bugs.

### 3. Verification-before-publish (not optional)
**Decision**: Citation coverage and entailment checks run before every publish. Reports that fail are marked `UNPUBLISHABLE`.
**Why**: An evidence engine that publishes unsupported claims is worse than useless — it's dangerous. The verification gate is the single most important design decision.
**Tradeoff**: ~48% of runs are marked unpublishable. That's honest, not a failure.

### 4. Multi-provider LLM with no fine-tuning
**Decision**: Use general-purpose LLMs through standardized APIs. No fine-tuning.
**Why**: (a) Reproducibility across providers matters more than marginal quality gains. (b) Fine-tuning creates model-specific coupling. (c) v0.1 should demonstrate the pipeline, not the model.
**Tradeoff**: Extraction quality depends entirely on prompt engineering and structured output parsing.

### 5. Fulltext via PMC Open Access only
**Decision**: Retrieve full text only from PMC Open Access subset (JATS XML).
**Why**: Free, legal, structured. Alternatives (Sci-Hub, institutional access) create legal/reproducibility issues.
**Tradeoff**: Many high-quality studies are not in PMC OA. This limits evidence depth.

### 6. SQLite for persistence (not Postgres)
**Decision**: Store run state, checkpoints, and screening decisions in SQLite.
**Why**: Zero-config, embedded, sufficient for single-user research workloads. Adding Postgres would require Docker orchestration for the simplest use case.
**Tradeoff**: No concurrent write support. Fine for v0.1.

### 7. Skeptic agent as structured critique
**Decision**: Implement the "Skeptic" as structured critique with severity-tagged findings, not a conversational debate.
**Why**: A structured critique (with dimensions like `internal_validity`, `overstatement`, `missing_evidence`) is testable and actionable. A free-form debate would be entertaining but unauditable.
**Tradeoff**: Less creative adversarial testing. Catches systematic issues but may miss subtle ones.

## Hard Problems Encountered

### 1. Citation Laundering
**Symptom**: LLMs would cite a study by PMID but the supporting text didn't come from that study.
**Root cause**: The LLM was "laundering" citations — generating plausible-sounding claims and attaching the most relevant-looking citation, regardless of whether the cited study actually supported the claim.
**Fix**: Mandatory snippet-level traceability. Every claim must link to a `snippet_id`, and every snippet has a verified `source_ref` pointing to a specific record and section. The verification node checks that cited snippets actually exist and were extracted from the cited study.
**Lesson**: You can't trust LLMs to cite honestly. You need structural enforcement.

### 2. Uniform RoB2 "Some Concerns"
**Symptom**: Nearly all RoB2 assessments came back as "some_concerns" across all 5 domains, regardless of study quality.
**Root cause**: Assessing bias from abstracts alone provides insufficient information. Abstracts rarely describe randomization procedures, blinding details, or dropout handling.
**Fix**: (a) Enable PMC fulltext retrieval by default. (b) When fulltext is available, prioritize Methods + Results sections for RoB2 assessment. (c) When only abstracts are available, explicitly note this limitation in the assessment rationale.
**Lesson**: Garbage in → garbage out. RoB2 on abstracts is structurally limited.

### 3. CT.gov Query Explosion
**Symptom**: ClinicalTrials.gov returned 0 results for most queries.
**Root cause**: The pipeline was passing full natural-language queries to CT.gov's API, which expects short keyword queries. Long queries hit API limits or returned empty sets.
**Fix**: Delegate query sanitization to the CT client (500-char limit with term extraction) instead of aggressive node-level truncation that destroyed meaning.
**Lesson**: Respect each API's query model. PubMed handles Boolean operators; CT.gov wants keywords.

### 4. PRISMA Counts Desync
**Symptom**: PRISMA flowchart numbers didn't add up (identified ≠ screened + excluded).
**Root cause**: Records were being modified in-place during screening, and deduplication counts were computed at the wrong pipeline stage.
**Fix**: Immutable records (Pydantic `frozen=True`), with screening decisions stored as separate `ScreeningDecision` objects linked by `record_id`. PRISMA counts computed once, atomically, in the publish node.
**Lesson**: Immutability isn't just a functional programming preference — it's a correctness requirement when counts need to be auditable.

### 5. Monolith Growth
**Symptom**: `graph.py` grew to ~3,000 lines with all 13 node functions inline.
**Root cause**: Incremental development without refactoring. Each audit iteration added logic to existing nodes.
**Fix**: Split into `orchestration/nodes/` package with 7 focused modules. `graph.py` reduced to ~500 lines (imports, conditional edges, graph builder, runner).
**Lesson**: Refactor continuously or pay the cost in one painful session.

## Evidence

See [EVAL.md](EVAL.md) for full evaluation methodology and results.

**Baseline summary (v0.1, 5-query golden set):**

| Metric | Target | Measured |
|--------|--------|----------|
| Citation coverage | ≥90% | Measured per-run |
| Claims with ≥1 snippet | 100% | Enforced by schema |
| Verification pass rate | ≥80% | Measured per-run |
| RoB2 domain coverage | 5/5 domains | Enforced by schema |
| Pipeline completion | ≥60% publishable | ~45% (honest) |
| Test suite | 0 failures | 635 backend + 81 frontend |

**Key takeaway**: CDR produces traceable, structured outputs. Evidence quality is limited by retrieval depth (PubMed + PMC OA only) and LLM extraction accuracy. This is documented, not hidden.

## What I'd Improve Next (v0.2)

1. **Embase/Cochrane integration**: The biggest retrieval gap. PubMed alone misses ~30% of relevant studies.
2. **GRADE automation**: Currently outputs certainty labels but doesn't implement the full GRADE framework (risk of bias, inconsistency, indirectness, imprecision, publication bias).
3. **Human-in-the-loop checkpoints**: Pause at screening/extraction for expert review before proceeding.
4. **Quantitative meta-analysis**: Currently qualitative synthesis only. Forest plots and pooled estimates would significantly increase utility.
5. **Authentication + multi-tenancy**: v0.1 has no auth. Required for any deployment beyond local research.
6. **Streaming output**: Real-time progress updates instead of waiting for the full pipeline to complete.

---

*This case study was written for CDR v0.1 Open Alpha. It reflects the actual state of the system, not aspirational goals.*

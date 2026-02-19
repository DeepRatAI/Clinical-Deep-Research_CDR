# CDR Incident Postmortems

> Lessons learned from real issues encountered during CDR development.
> These are documented for transparency and to help future contributors.

---

## INC-001: Citation Laundering

**Date**: December 2025
**Severity**: Critical
**Status**: Resolved

### What happened

During evaluation of GS-001 (aspirin for secondary CV prevention), we discovered that several claims cited PMIDs that did not actually support the claim text. The LLM was generating plausible claims and attaching the most "relevant-looking" PMID from the retrieved set, without verifying that the cited study actually contained the supporting evidence.

**Example**: A claim about "21% risk reduction in MACE" cited PMID 12345678, but that study measured stroke incidence, not MACE.

### Impact

- 3 out of 8 claims in a test run had incorrect citations
- Report appeared authoritative but contained misleading source attributions
- This is the most dangerous failure mode for an evidence engine

### Root cause

The extraction → synthesis pipeline did not enforce snippet-level provenance. Claims were generated from the LLM's "understanding" of the evidence corpus, then citations were attached as a post-hoc decoration rather than being structurally linked during extraction.

### Fix

1. **Mandatory snippet IDs**: Every `EvidenceClaim` now requires `supporting_snippet_ids` (enforced by Pydantic `min_length=1`)
2. **Snippet traceability**: Every `Snippet` has a `source_ref` pointing to a specific `Record` and `Section`
3. **Verification node**: Added citation coverage check that verifies each cited snippet actually exists and was extracted from the cited study
4. **Unpublishable gate**: Reports with citation coverage < 90% are marked `unpublishable`

### Lessons

- **Never trust LLMs to cite honestly.** Structural enforcement is the only reliable approach.
- **Citation accuracy is a safety property**, not a quality metric. Treat it like a correctness invariant.
- The Pydantic schema is the last line of defense — if the schema allows it, it will happen.

---

## INC-002: Uniform RoB2 "Some Concerns"

**Date**: January 2026
**Severity**: High
**Status**: Mitigated (structural limitation remains)

### What happened

Across 50+ evaluation runs, nearly every RoB2 assessment returned "some_concerns" for all 5 domains. High-quality RCTs (large, well-conducted) received the same rating as small pilot studies. The RoB2 node was producing uniform, uninformative output.

### Impact

- RoB2 information was present but useless — no discrimination between study qualities
- Downstream synthesis could not properly weight evidence by quality
- GRADE certainty assessments were artificially uniform

### Root cause

RoB2 assessment was being performed on **abstracts only**. Abstracts rarely contain the information needed for proper bias assessment:
- Randomization procedures (domain 1) are almost never described in abstracts
- Blinding details (domain 2) are mentioned briefly if at all
- Dropout rates (domain 3) are not in abstracts
- Outcome measurement details (domain 4) are abbreviated

The LLM correctly assessed "insufficient information → some concerns" for each domain, which is technically the right answer but practically useless.

### Fix

1. **PMC fulltext retrieval**: Enabled by default for all records with PMC OA availability
2. **Methods prioritization**: When fulltext is available, the RoB2 node now prioritizes Methods + Results sections
3. **Explicit limitation**: When only abstracts are available, each domain rationale now includes: "Assessment limited to abstract; full text not available"
4. **Quality flag**: Added `fulltext_available` boolean to study cards so downstream consumers know the assessment basis

### Lessons

- **Garbage in → garbage out.** The most sophisticated assessment framework is useless with insufficient input.
- **Fail transparently.** Reporting "insufficient information" is more honest than inventing a judgment.
- Fulltext access is the single biggest quality lever for RoB2. Institutions with Elsevier/Springer access would see dramatically better results.

---

## INC-003: PRISMA Count Arithmetic Failures

**Date**: January 2026
**Severity**: Medium
**Status**: Resolved

### What happened

PRISMA flowchart numbers in published reports frequently didn't add up:
- `records_identified` ≠ `records_from_pubmed` + `records_from_clinical_trials` + `records_from_other`
- `records_screened` ≠ `records_identified` - `duplicates_removed`
- `reports_assessed` + `reports_excluded` ≠ `reports_sought`

### Impact

- PRISMA compliance was broken — the flowchart was decorative, not auditable
- Reviewers familiar with PRISMA would immediately distrust the output
- Counts were computed at different points in the pipeline, leading to race conditions

### Root cause

Two interacting issues:

1. **Mutable records**: Records were Python dicts modified in-place during screening. When a record was excluded, its state changed, but earlier count snapshots weren't updated.

2. **Distributed counting**: Each node computed its own partial counts. The publish node tried to reconcile them, but different nodes had different views of the data at different pipeline stages.

### Fix

1. **Immutable records**: All `Record` models now use `frozen=True` (Pydantic). Screening decisions are stored as separate `ScreeningDecision` objects linked by `record_id`.

2. **Atomic count computation**: PRISMA counts are computed exactly once, in the publish node, from the final immutable state. No intermediate count snapshots.

3. **Arithmetic validators**: Added `model_validator` on `PRISMACounts` that checks:
   - `records_identified == records_from_pubmed + records_from_clinical_trials + records_from_other`
   - `records_screened == records_identified - duplicates_removed`
   - `studies_included <= reports_assessed`

4. **Test coverage**: Added `test_prisma_counts_arithmetic` that verifies all invariants.

### Lessons

- **Immutability isn't academic** — it's a correctness requirement when counts need to be auditable.
- **Compute once, at the end.** Don't snapshot intermediate counts if the final state is available.
- PRISMA compliance is a **structural property** of the pipeline, not something you can bolt on at the end.

---

*These postmortems are part of CDR's commitment to honest reporting. If you encounter a new failure mode, please document it following this template and submit a PR.*

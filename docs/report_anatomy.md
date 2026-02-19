# CDR Report Anatomy

> **How to read and audit a CDR evidence report**

---

## Overview

A CDR report is a **structured JSON document** containing the complete audit trail of a clinical research query: from the original question through evidence retrieval, screening, synthesis, and final assertions.

Each report is:
- **Traceable**: Every claim links to source snippets (PMID, page, quote)
- **Reviewable**: Risk of Bias assessments for each study
- **Verifiable**: Matches PRISMA guidelines for systematic reviews
- **Stamped**: Includes disclaimer and metadata (run_id, timestamp)

---

## Field-by-Field Guide

### 1. **Metadata Fields**

#### `run_id` (string)
Unique identifier for this execution. Enables reproduction and audit trail.

```json
{
  "run_id": "sample-aspirin-001",
  ...
}
```

**How to use**: Reference this ID when asking "What are the conditions of this run?" It's stable across reruns of the same input.

#### `question` (string)
The original clinical research question, exactly as submitted.

```json
{
  "question": "Is aspirin effective for secondary prevention of cardiovascular events in adults?"
}
```

#### `timestamp` (ISO 8601 datetime)
When the pipeline executed. Useful for understanding when evidence was retrieved (PubMed index date matters).

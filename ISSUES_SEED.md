# CDR Starter Issues

> 10 curated issues for new contributors. Mix of difficulty levels.

---

## Good First Issues

### 1. 📝 Add type annotations to `eval/eval_runner.py`

**Labels**: `good-first-issue`, `code-quality`

The eval runner uses basic Python types. Add full type annotations (function params, return types, key variables) and verify with `mypy`.

**Files**: `eval/eval_runner.py`
**Effort**: Small (1-2 hours)

---

### 2. 📝 Improve error messages in `.env` validation

**Labels**: `good-first-issue`, `dx`

When required environment variables are missing, CDR throws generic `KeyError`. Add a startup validator that checks required vars and prints a helpful message pointing to `.env.example`.

**Files**: `src/cdr/core/config.py` (or new file)
**Effort**: Small (2-3 hours)

---

### 3. 📝 Add `--format` flag to evaluation runner

**Labels**: `good-first-issue`, `eval`

The eval runner outputs JSON. Add a `--format` flag that supports `json` (default) and `markdown` (human-readable table).

**Files**: `eval/eval_runner.py`
**Effort**: Small (2-3 hours)

---

## Docs / Infrastructure

### 4. 📗 Document all API endpoints with examples

**Labels**: `docs`, `api`

The FastAPI auto-docs at `/docs` exist but lack usage examples. Add example request/response pairs for the 5 most important endpoints: `/api/v1/run`, `/api/v1/runs/{id}`, `/api/v1/runs/{id}/report`, `/api/v1/export/{id}`, `/api/v1/health`.

**Files**: `docs/api-reference.md`, route docstrings in `src/cdr/api/routes.py`
**Effort**: Medium (3-4 hours)

---

### 5. 📗 Add Dependabot configuration

**Labels**: `docs`, `infrastructure`

Set up `.github/dependabot.yml` for automated dependency update PRs. Configure for both pip (Python) and npm (frontend). Set weekly schedule and group minor updates.

**Files**: `.github/dependabot.yml`
**Effort**: Small (1 hour)

---

### 6. 📗 Create architecture diagram as SVG

**Labels**: `docs`, `visualization`

The README has an ASCII architecture diagram. Create a proper SVG version (using Mermaid, diagrams.net, or similar) showing the 13-node pipeline with data flow types.

**Files**: `docs/images/architecture.svg`, update `README.md`
**Effort**: Medium (3-4 hours)

---

## Evaluation / Quality

### 7. 📊 Add latency tracking to evaluation runs

**Labels**: `eval`, `observability`

Track and report wall-clock time per pipeline stage during evaluation. Output p50 and p95 latencies in the evaluation summary.

**Files**: `eval/eval_runner.py`, `src/cdr/evaluation/metrics.py`
**Effort**: Medium (4-5 hours)

---

### 8. 📊 Implement `context_precision` metric

**Labels**: `eval`, `enhancement`

The metrics module defines `context_precision` but doesn't implement it fully. Implement it as: (number of retrieved snippets that appear in claims) / (total retrieved snippets). Add tests.

**Files**: `src/cdr/evaluation/metrics.py`, `tests/test_evaluation_metrics.py`
**Effort**: Medium (3-4 hours)

---

## Bug / Tech Debt

### 9. 🐛 CT.gov client returns empty for long queries

**Labels**: `bug`, `retrieval`

When the planned query exceeds ~500 characters, the CT.gov API returns 0 results. The client should truncate/extract keywords more aggressively. Add a test that verifies behavior with a 600-char query.

**Files**: `src/cdr/retrieval/ctgov_client.py`, `tests/test_ctgov_client.py`
**Effort**: Medium (3-4 hours)
**Hint**: See [INC-003 in INCIDENTS.md](INCIDENTS.md) for context.

---

### 10. 🐛 Publisher HTML template lacks mobile responsiveness

**Labels**: `bug`, `frontend`

The HTML report template uses fixed widths. Add basic responsive CSS so reports are readable on mobile. Test with Chrome DevTools responsive mode.

**Files**: `src/cdr/publisher/templates/` (or inline HTML in `publisher.py`)
**Effort**: Small (2-3 hours)

---

## How to Claim an Issue

1. Comment on the GitHub issue "I'd like to work on this"
2. Fork the repo and create a branch (`feat/issue-N-description`)
3. Implement, test, submit PR
4. Reference the issue number in your PR description

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development setup.

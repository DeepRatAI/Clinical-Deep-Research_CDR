# CDR v0.1 Open Alpha — Release Criteria

## Definition of Done

CDR v0.1 Open Alpha is considered **done** when ALL of the following are true:

### 1. Executable by Third Parties

- [ ] A new user can clone the repo and run CDR end-to-end with `make setup && make server`
- [ ] `make test` passes with 0 failures
- [ ] `.env.example` documents all required and optional variables
- [ ] Quickstart in README covers local, Docker, and manual paths

### 2. Output Stable and Auditable

- [ ] Report output follows `schemas/report.schema.json`
- [ ] Every claim has `supporting_snippet_ids` (enforced by Pydantic schema)
- [ ] Clinical disclaimer appears in every report output
- [ ] Reports include PRISMA counts, RoB2 summary, and verification status

### 3. Evidence Reproducible

- [ ] `eval/` contains evaluation runner, dataset, and baseline results
- [ ] `make eval` runs evaluation and produces comparable results
- [ ] Seed/determinism documented in EVAL.md
- [ ] Baseline results versioned in `eval/results/baseline_v0_1.json`

### 4. Observability Minimum

- [ ] Structured logging with `run_id` + node name at each pipeline stage
- [ ] Metrics collection (counters, timings) via `observability/metrics.py`
- [ ] Errors logged with full context (node, run_id, traceback)

### 5. Open-Source Ready

- [ ] Apache 2.0 LICENSE file present
- [ ] DISCLAIMER.md with clinical limitations
- [ ] SECURITY.md with vulnerability reporting process
- [ ] CONTRIBUTING.md with setup, test, and PR guidelines
- [ ] `.github/ISSUE_TEMPLATE/` with bug and feature templates
- [ ] `.gitignore` covers all generated artifacts
- [ ] `requirements.lock` for pinned dependencies
- [ ] No secrets in version control

### 6. Quality Minimum

- [ ] 635+ backend tests passing
- [ ] 81 frontend tests passing
- [ ] CI workflow runs lint + tests + coverage
- [ ] Stage-level contracts documented in `docs/contracts/`
- [ ] Report output schema in `schemas/report.schema.json`

### 7. Documentation Complete

- [ ] README.md with quickstart, architecture, troubleshooting
- [ ] CASE_STUDY.md with decisions, tradeoffs, hard problems
- [ ] EVAL.md with methodology and results
- [ ] ROADMAP.md with v0.1 / v0.2 / v1.0
- [ ] CHANGELOG.md with v0.1.0 entry
- [ ] INCIDENTS.md with 3 postmortem entries
- [ ] ISSUES_SEED.md with 10 starter issues

## Verification

Run the following to verify all criteria:

```bash
# Tests pass
make test
make test-ui

# Evaluation runs
make eval

# Lint passes
make lint

# All docs exist
for f in README.md CASE_STUDY.md EVAL.md DISCLAIMER.md CONTRIBUTING.md \
         ROADMAP.md SECURITY.md CHANGELOG.md RELEASE_CRITERIA.md \
         INCIDENTS.md ISSUES_SEED.md LICENSE .gitignore requirements.lock; do
  [ -f "$f" ] && echo "✅ $f" || echo "❌ $f MISSING"
done
```

## Sign-off

| Criterion | Status |
|-----------|--------|
| Executable by third parties | ⬜ |
| Output stable and auditable | ⬜ |
| Evidence reproducible | ⬜ |
| Observability minimum | ⬜ |
| Open-source ready | ⬜ |
| Quality minimum | ⬜ |
| Documentation complete | ⬜ |

# CDR v0.1 Open Alpha — Launch Checklist

> Final verification before publishing.

## Pre-Launch Checklist

### Documentation

| # | Item | File | Status |
|---|------|------|--------|
| 1 | README with quickstart, arch, troubleshooting | README.md | ✅ |
| 2 | Clinical case study with decisions + tradeoffs | CASE_STUDY.md | ✅ |
| 3 | Evaluation methodology + baseline | EVAL.md | ✅ |
| 4 | Clinical disclaimer | DISCLAIMER.md | ✅ |
| 5 | Contribution guide | CONTRIBUTING.md | ✅ |
| 6 | Roadmap (v0.1 → v0.2 → v1.0) | ROADMAP.md | ✅ |
| 7 | Security policy | SECURITY.md | ✅ |
| 8 | Changelog with v0.1.0 entry | CHANGELOG.md | ✅ |
| 9 | Release criteria | RELEASE_CRITERIA.md | ✅ |
| 10 | Incident postmortems (3 entries) | INCIDENTS.md | ✅ |
| 11 | Starter issues (10 curated) | ISSUES_SEED.md | ✅ |
| 12 | Pipeline contracts (13 stages) | docs/contracts/pipeline_contracts.md | ✅ |

### Legal

| # | Item | File | Status |
|---|------|------|--------|
| 13 | Apache 2.0 license | LICENSE | ✅ |
| 14 | License in pyproject.toml | pyproject.toml | ✅ |
| 15 | No AGPL references remaining | — | ✅ Verified |

### Engineering

| # | Item | File | Status |
|---|------|------|--------|
| 16 | .gitignore covers all artifacts | .gitignore | ✅ |
| 17 | uv.lock pinned (reproducible installs) | uv.lock | ✅ |
| 18 | Makefile with all advertised targets | Makefile | ✅ |
| 19 | .env.example complete | .env.example | ✅ |
| 20 | Report JSON Schema | schemas/report.schema.json | ✅ |
| 21 | CI workflow functional | .github/workflows/ci.yml | ✅ |
| 22 | Issue templates (bug, feature, docs) | .github/ISSUE_TEMPLATE/ | ✅ |

### Evaluation

| # | Item | File | Status |
|---|------|------|--------|
| 23 | Eval runner | eval/eval_runner.py | ✅ |
| 24 | Golden set dataset | eval/datasets/golden_set_toy.json | ✅ |
| 25 | Baseline results | eval/results/baseline_v0_1.json | ✅ |

### Examples

| # | Item | File | Status |
|---|------|------|--------|
| 26 | Python query example | examples/run_query.py | ✅ |
| 27 | Evaluation example | examples/evaluate.py | ✅ |

### Tests

| # | Item | Status |
|---|------|--------|
| 28 | Backend tests pass (635) | ✅ Verified |
| 29 | Frontend tests pass (81) | ⬜ Verify (`cd ui && npm ci && npm test -- --run`) |
| 30 | No secrets in codebase | ✅ Verified |

## Release Steps

```bash
# 1. Install dependencies (reproducible)
make sync

# 2. Verify all tests pass
make test
make test-ui

# 3. Verify lint passes
make lint

# 4. Verify all docs exist
for f in README.md CASE_STUDY.md EVAL.md DISCLAIMER.md CONTRIBUTING.md \
         ROADMAP.md SECURITY.md CHANGELOG.md RELEASE_CRITERIA.md \
         INCIDENTS.md ISSUES_SEED.md LICENSE .gitignore uv.lock; do
  [ -f "$f" ] && echo "✅ $f" || echo "❌ $f MISSING"
done

# 4. Verify no secrets
grep -rn "sk-" --include="*.py" src/ tests/ || echo "✅ No OpenAI keys"
grep -rn "hf_" --include="*.py" src/ tests/ || echo "✅ No HF tokens"

# 5. Verify no AGPL references
grep -rn "AGPL" --include="*.py" --include="*.toml" --include="*.md" . || echo "✅ No AGPL refs"

# 6. Tag release
git add -A
git commit -m "release: CDR v0.1.0 Open Alpha"
git tag -a v0.1.0 -m "CDR v0.1 Open Alpha — first public release"
git push origin main --tags

# 7. Create GitHub Release
# Title: CDR v0.1.0 — Open Alpha
# Body: See CHANGELOG.md for details
```

## Post-Release

- [ ] Create GitHub Issues from ISSUES_SEED.md
- [ ] Verify GitHub Actions CI runs on the release
- [ ] Test clean clone + setup on a fresh machine
- [ ] Announce on relevant channels

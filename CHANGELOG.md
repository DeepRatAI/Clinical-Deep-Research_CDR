# Changelog

All notable changes to CDR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`--format` flag for evaluation runner**: `eval_runner.py` now accepts `--format json|markdown|all` to control output file generation. Default is `all` (backward-compatible). Closes [#3](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/issues/3). ([#24](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/24))
- **Dependabot configuration**: Automated dependency update monitoring for pip, npm, and GitHub Actions on a weekly Monday schedule. Closes [#5](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/issues/5).
- **Pull Request template**: Standardized PR template (`.github/PULL_REQUEST_TEMPLATE.md`) with checklist for description, tests, docs, and breaking changes.
- **Code of Conduct**: Contributor Covenant v2.1 (`CODE_OF_CONDUCT.md`).
- **Branch protection on `main`**: Require 1 PR review, CDR CI status check, dismiss stale reviews, no force pushes or deletions.

### Changed

- **Comprehensive type annotations in `eval_runner.py`**: Added 7 type aliases (`EvalMode`, `OutputFormat`, `QuestionDict`, `MetricsDict`, `EvalResult`, `EvalSummary`, `ComparisonResult`), explicit variable annotations, and full `typing` imports. Passes `mypy --strict` with 0 errors. Closes [#1](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/issues/1). ([#25](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/25))

### Dependencies

- **GitHub Actions** (CI):
  - `actions/checkout` 4 → 6 ([#13](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/13))
  - `actions/upload-artifact` 4 → 6 ([#14](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/14))
  - `actions/setup-python` 5 → 6 ([#15](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/15))
  - `actions/cache` 4 → 5 ([#16](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/16))
  - `actions/setup-node` 4 → 6 ([#17](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/17))
- **npm** (`/ui`):
  - Minor-and-patch group: 7 updates including vite, vitest, eslint plugins ([#18](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/18))
  - `jsdom` 25.0.1 → 28.1.0 ([#23](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/23))

### Removed

- Closed Dependabot PRs for incompatible major bumps: tailwindcss 3→4 ([#19](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/19)), eslint 9→10 ([#20](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/20)), react-dom ([#22](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/pull/22)). These require migration work and may be addressed in v0.2.0.

### Community

- Community Profile score raised to 8/8.
- Welcome Discussion created for new contributors.
- Detailed code review posted on [#2](https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/issues/2) for Tianlin0725's environment validation implementation.

## [0.1.0] - 2026-02-16

### 🎉 Initial Open Alpha Release

First public release of Clinical Deep Research (CDR).

### Added

- **13-node LangGraph pipeline**: parse_question → plan_search → retrieve → deduplicate → screen → parse_docs → extract_data → assess_rob2 → synthesize → critique → verify → compose → publish
- **Multi-source retrieval**: PubMed E-utilities + ClinicalTrials.gov v2 API
- **Hybrid ranking**: BM25 sparse + sentence-transformer dense + cross-encoder reranking
- **PMC Open Access fulltext**: JATS XML parsing with structured section extraction
- **Screening**: LLM-based inclusion/exclusion with PICO matching and reason codes
- **Structured extraction**: StudyCards via DSPy-based extraction
- **RoB2 assessment**: All 5 Cochrane domains with Methods section prioritization
- **ROBINS-I assessment**: For non-randomized studies
- **Compositional inference**: A+B⇒C hypothesis generation from cross-study evidence
- **Adversarial critique**: Skeptic agent with 8 critique dimensions
- **Verification**: Citation coverage + entailment checking with quality gates
- **Multi-format output**: Markdown, JSON, HTML reports with PRISMA flowchart
- **8 LLM providers**: Gemini, OpenAI, Anthropic, HuggingFace, Groq, Cerebras, OpenRouter, Cloudflare
- **FastAPI server**: 18 endpoints including run management, export, and metrics
- **React frontend**: Basic UI for run management and report viewing
- **SQLite persistence**: RunStore with checkpoints, records, screening decisions
- **Observability**: Structured tracing (request_id + stage) and metrics collection
- **Test suite**: 635 backend + 81 frontend tests, 0 failures
- **Evaluation framework**: 5-query golden set with 7 quality metrics
- **CI/CD**: GitHub Actions with lint, test, coverage, OpenAPI validation

### Security

- Apache 2.0 license
- Clinical disclaimer on all outputs
- No secrets in repository
- Pinned dependencies (`requirements.lock`)

### Known Issues

- ~45% of runs produce publishable reports; ~48% are marked unpublishable (honest reporting)
- Retrieval limited to PubMed + CT.gov (no Embase/Cochrane)
- No authentication in API
- Fulltext available only for PMC Open Access subset
- RoB2 quality depends on fulltext availability

[Unreleased]: https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/releases/tag/v0.1.0

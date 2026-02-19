# Changelog

All notable changes to CDR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/DeepRatAI/Clinical-Deep-Research_CDR/releases/tag/v0.1.0

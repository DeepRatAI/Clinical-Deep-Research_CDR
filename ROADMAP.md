# CDR Roadmap

## v0.1 Open Alpha (current)

**Status**: Released — February 2026

The minimum viable evidence engine: a working pipeline that retrieves literature, screens, extracts, assesses bias, synthesizes, verifies, and publishes — with full traceability.

**What's included:**
- 13-node LangGraph pipeline (parse → plan → retrieve → screen → extract → assess → synthesize → critique → verify → compose → publish)
- PubMed + ClinicalTrials.gov retrieval with BM25 + dense + reranking
- PMC Open Access fulltext retrieval (JATS XML)
- RoB2 (5 domains) + ROBINS-I bias assessment
- Verification gates (citation coverage, entailment)
- Structured output: JSON + Markdown + HTML
- 8 LLM providers (Gemini, OpenAI, Anthropic, HuggingFace, Groq, Cerebras, OpenRouter, Cloudflare)
- FastAPI with 18 endpoints
- React frontend (basic)
- 716 tests (635 backend + 81 frontend), 0 failures
- Evaluation baseline with 5-query golden set
- Apache 2.0 license

**Known limitations:**
- Single-database retrieval (no Embase/Cochrane)
- Qualitative synthesis only (no meta-analysis)
- No authentication
- No streaming
- ~45% publishable rate (honest, not a bug)

---

## v0.2 (next)

**Focus**: Evidence depth + usability

| Feature | Priority | Description |
|---------|----------|-------------|
| GRADE framework | High | Full GRADE certainty assessment (risk of bias, inconsistency, indirectness, imprecision, publication bias) |
| Human-in-the-loop | High | Checkpoint at screening/extraction for expert review |
| Streaming output | High | Real-time pipeline progress via SSE/WebSocket |
| Embase integration | Medium | Requires institutional API access; doubles retrieval coverage |
| Authentication | Medium | JWT/OAuth2 for multi-user deployments |
| Quantitative synthesis | Medium | Forest plots, pooled effect estimates, heterogeneity metrics |
| PDF upload | Medium | User-supplied papers as additional evidence sources |
| Improved UI | Medium | PRISMA flow visualization, study card browser, evidence explorer |
| Cochrane integration | Low | CENTRAL database access |
| Multi-language | Low | Support non-English literature retrieval |

---

## v1.0 (vision)

**Focus**: Research-grade systematic review assistant

- Full PRISMA 2020 compliance (all 27 items)
- Multi-database retrieval (PubMed + Embase + Cochrane + grey literature)
- Quantitative meta-analysis with sensitivity analysis
- Living review capability (scheduled re-runs with diff)
- Collaborative review (multiple reviewers, consensus)
- Institutional deployment (SSO, audit logging, data governance)
- Validated against expert-conducted systematic reviews
- Published evaluation in peer-reviewed venue

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help, and [ISSUES_SEED.md](ISSUES_SEED.md) for starter issues.

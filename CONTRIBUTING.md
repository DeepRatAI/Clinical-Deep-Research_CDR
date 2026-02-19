# Contributing to CDR

Thank you for your interest in contributing to Clinical Deep Research!

CDR is an open alpha — we welcome bug reports, documentation improvements, test additions, and feature proposals.

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/DeepRatAI/Clinical-Deep-Research_CDR.git
cd Clinical-Deep-Research_CDR
make setup
source .venv/bin/activate
cp .env.example .env
# Edit .env with at least HF_TOKEN
```

### 2. Run tests

```bash
make test          # Backend (635 tests)
make test-ui       # Frontend (81 tests)
make lint          # Ruff + mypy
```

### 3. Run the server

```bash
make dev           # API at http://localhost:8000/docs
```

## What Can I Work On?

See [ISSUES_SEED.md](ISSUES_SEED.md) for starter issues, or check the GitHub Issues tab.

**Labels:**
- `good-first-issue` — Suitable for first-time contributors
- `docs` — Documentation improvements
- `eval` — Evaluation and metrics
- `bug` — Confirmed bugs
- `enhancement` — Feature proposals

## Development Workflow

### Branch naming

```
feat/short-description    # New feature
fix/short-description     # Bug fix
docs/short-description    # Documentation
test/short-description    # Test additions
```

### Making changes

1. Create a branch from `main`
2. Make your changes
3. Run `make test && make lint` (must pass)
4. Commit with a descriptive message (see below)
5. Open a Pull Request

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Embase retrieval client
fix: correct PRISMA count desync in publish node
docs: improve quickstart for Docker path
test: add RoB2 domain coverage tests
refactor: split extract_data node into focused modules
```

### Pull Request checklist

- [ ] Tests pass (`make test && make lint`)
- [ ] New code has tests
- [ ] Docstrings on public functions
- [ ] CHANGELOG.md updated (for user-visible changes)
- [ ] No secrets or credentials in code
- [ ] Clinical disclaimer preserved in any output changes

## Code Style

- **Python**: Ruff + Black formatting, Google-style docstrings
- **Type hints**: Required on all public functions
- **Pydantic**: All pipeline data flows through typed schemas
- **Tests**: pytest, async tests with `asyncio_mode = "auto"`

### Import order

```python
# stdlib
import json
from pathlib import Path

# third-party
from pydantic import BaseModel
import httpx

# local
from cdr.core.schemas import EvidenceClaim
from cdr.core.enums import GRADECertainty
```

## Adding a Pipeline Node

CDR's pipeline is a LangGraph StateGraph with 13 nodes. To add or modify a node:

1. **Schema first**: Define input/output types in `src/cdr/core/schemas.py`
2. **Implementation**: Add node function in `src/cdr/orchestration/nodes/`
3. **Registration**: Wire into the graph in `src/cdr/orchestration/graph.py`
4. **Contract**: Document I/O in `docs/contracts/`
5. **Tests**: Add unit tests in `tests/`

### Key invariants to preserve

- Every claim must have `supporting_snippet_ids` (enforced by Pydantic)
- Every exclusion must have a `reason` code
- PRISMA counts must be internally consistent
- Verification runs before publish (never skip)

## Adding an LLM Provider

CDR supports multiple LLM providers. To add a new one:

1. Add client in `src/cdr/llm/providers/`
2. Register in `src/cdr/llm/factory.py`
3. Add env vars to `.env.example`
4. Add tests in `tests/test_llm_*.py`

## Testing Guidelines

- **Unit tests**: Test individual functions with mocked dependencies
- **Integration tests**: Test node-to-node data flow
- **Golden set tests**: Validate against known clinical questions
- Don't test LLM output content (non-deterministic) — test structure and contracts

### Running specific tests

```bash
# Filter tests by name
make test ARGS="-k test_rag_service"

# Run a single file
make test ARGS="tests/test_chunking.py -s"

# With coverage
PYTHONPATH=src pytest tests/ --cov=src/cdr --cov-report=term-missing
```

## Documentation

- **README.md**: User-facing, quickstart + overview
- **CASE_STUDY.md**: Technical narrative, decisions, tradeoffs
- **EVAL.md**: Evaluation methodology and results
- **docs/contracts/**: Per-stage I/O specifications
- **Docstrings**: Required on all public classes and functions

## Clinical Content Guidelines

CDR is a research tool, not a medical device. When writing documentation or output templates:

1. **Never claim** CDR provides medical advice
2. **Always include** the clinical disclaimer in outputs
3. **Use hedging language**: "evidence suggests" not "evidence proves"
4. **Report limitations honestly**: missing databases, LLM uncertainty
5. **Preserve traceability**: every claim → snippet → source

## Questions?

- Open a GitHub Issue for bugs or feature requests
- Check [CASE_STUDY.md](CASE_STUDY.md) for architectural context
- See [ROADMAP.md](ROADMAP.md) for planned features

---

*Thank you for helping make clinical evidence more accessible.*

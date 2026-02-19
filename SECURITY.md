# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please report them privately:

1. **Email**: security@deepratai.com
2. **Subject**: `[CDR Security] <brief description>`
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within **48 hours** and aim to provide a fix or mitigation within **7 days** for critical issues.

## Security Practices

### Secrets Management

- **Never commit API keys, tokens, or credentials** to the repository
- Use `.env` files (excluded via `.gitignore`) for local secrets
- CI/CD secrets are stored in GitHub Actions encrypted secrets
- The `.env.example` file contains only placeholder values

### Dependency Management

- Dependencies are declared in `pyproject.toml` with minimum version pins
- A pinned `requirements.lock` is provided for reproducible builds
- Run `pip audit` periodically to check for known vulnerabilities
- Frontend dependencies use `npm audit` in CI

### Data Handling

- CDR retrieves data from **public APIs** (PubMed, ClinicalTrials.gov)
- No patient data, PHI, or PII is processed or stored
- Local SQLite storage contains only run metadata and retrieved abstracts
- Report outputs may contain excerpts from published literature (fair use)

### Network Security

- All external API calls use HTTPS
- Tests are fully mocked — no network calls in the test suite
- The API server binds to `0.0.0.0:8000` by default; restrict in production

### Supply Chain

- All Python dependencies are from PyPI
- Frontend dependencies are from npm
- No vendored binaries or pre-built artifacts
- Dockerfile uses official `python:3.12-slim` base image

## Threat Model (v0.1 Alpha)

| Threat | Mitigation | Status |
|--------|------------|--------|
| LLM prompt injection via user query | Input sanitization + structured extraction | Partial |
| API key leakage in outputs | Keys never enter the pipeline state | Implemented |
| Malicious PDF in retrieval | PDFs parsed in sandboxed parser (PyMuPDF) | Partial |
| Supply chain attack via deps | Pinned deps + `pip audit` | Implemented |
| Unauthorized API access | No auth in v0.1 (research tool) | TODO for v0.2 |

## Acknowledgments

We appreciate responsible disclosure. Contributors who report valid security issues will be credited in release notes (with permission).

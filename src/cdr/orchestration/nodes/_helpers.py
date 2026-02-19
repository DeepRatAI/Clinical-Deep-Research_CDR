"""
PICO Matching Helpers

Heuristic helpers for PICO-based screening when LLM is not available.
Used by screen_node for Level 1 (exploratory) fallback screening.

Refs: PRISMA 2020, Cochrane Handbook Section 4.6
"""

from __future__ import annotations

from cdr.core.schemas import PICO


def _extract_pico_terms(pico: PICO) -> dict[str, list[str]]:
    """Extract searchable terms from PICO for heuristic matching.

    Returns dict with keys: population, intervention, comparator, outcome
    Each value is a list of lowercase terms extracted from the PICO field.

    Refs: PRISMA 2020, Cochrane Handbook Section 4.6
    """

    def tokenize(text: str) -> list[str]:
        """Simple tokenization - split on whitespace and common separators."""
        if not text:
            return []
        # Normalize and split
        import re

        tokens = re.split(r"[\s,;/()]+", text.lower())
        # Filter short tokens and common stopwords
        stopwords = {"the", "a", "an", "and", "or", "with", "in", "of", "for", "to"}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    return {
        "population": tokenize(pico.population),
        "intervention": tokenize(pico.intervention),
        "comparator": tokenize(pico.comparator) if pico.comparator else [],
        "outcome": tokenize(pico.outcome),
    }


def _calculate_pico_match_score(text: str, pico_terms: dict[str, list[str]]) -> tuple[float, int]:
    """Calculate PICO match score based on term presence.

    Score breakdown:
    - Population: 0.3 if any term matches
    - Intervention: 0.3 if any term matches
    - Outcome: 0.3 if any term matches
    - Comparator: 0.1 if any term matches (optional component)

    Returns: tuple of (score: float 0.0-1.0, components_matched: int 0-4)

    CRITICAL: Per PRISMA 2020, at least 2 core PICO components (P+I or P+O)
    should match for meaningful inclusion. Single-component matches are weak.

    Refs: PRISMA 2020 eligibility criteria, Cochrane Handbook Section 4.6
    """
    text_lower = text.lower()
    score = 0.0
    components_matched = 0

    # Weight each PICO component
    weights = {
        "population": 0.3,
        "intervention": 0.3,
        "outcome": 0.3,
        "comparator": 0.1,
    }

    for component, weight in weights.items():
        terms = pico_terms.get(component, [])
        if any(term in text_lower for term in terms):
            score += weight
            components_matched += 1

    return score, components_matched

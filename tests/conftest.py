"""
CDR Test Configuration

Shared fixtures and test utilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# Set test environment before any imports
os.environ.setdefault("CDR_DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# Use temp directories for storage during tests (avoid /data permission issues)
_test_temp_dir = Path(tempfile.gettempdir()) / "cdr_test"
_test_temp_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CDR_ARTIFACT_PATH", str(_test_temp_dir / "artifacts"))
os.environ.setdefault("CDR_DB_PATH", str(_test_temp_dir / "cdr_test.db"))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def golden_set_dir() -> Path:
    """Return the golden set directory."""
    return Path(__file__).parent / "golden_set"


@pytest.fixture(autouse=True)
def reset_settings() -> Generator[None, None, None]:
    """Reset settings before each test."""
    from cdr.config import reset_settings

    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def sample_pico() -> dict:
    """Sample PICO for testing."""
    return {
        "population": "Adults with type 2 diabetes mellitus",
        "intervention": "GLP-1 receptor agonists",
        "comparator": "Placebo or standard care",
        "outcome": "HbA1c reduction and cardiovascular events",
        "study_types": ["rct", "meta_analysis"],
    }


@pytest.fixture
def sample_record() -> dict:
    """Sample evidence record for testing."""
    return {
        "record_id": "record_001",  # Correct field name
        "source": "pubmed",  # Lowercase enum value
        "content_hash": "abc123def456",  # Required field
        "external_id": "PMID:12345678",
        "title": "Effect of GLP-1 Agonists on HbA1c in Type 2 Diabetes",
        "abstract": "Background: Type 2 diabetes requires effective glycemic control. "
        "Methods: We conducted a randomized controlled trial. "
        "Results: HbA1c decreased by 1.2% in the treatment group. "
        "Conclusions: GLP-1 agonists are effective.",
        "authors": ["Smith J", "Doe A"],
        "year": 2023,
        "journal": "Diabetes Care",
        "doi": "10.2337/dc23-0001",
        "publication_type": ["Randomized Controlled Trial"],  # Correct field
        "metadata": {"impact_factor": 19.1},
    }


@pytest.fixture
def sample_snippet() -> dict:
    """Sample evidence snippet for testing."""
    return {
        "snippet_id": "snip_001",  # Correct field name
        "text": "HbA1c decreased by 1.2% (95% CI: 0.9-1.5, p<0.001) in the GLP-1 group compared to placebo.",
        "source_ref": {
            "record_id": "record_001",
            "pmid": "12345678",
            "section": "abstract",  # Lowercase enum value
        },
        "section": "abstract",  # Lowercase enum value
    }


@pytest.fixture
def sample_study_card() -> dict:
    """Sample StudyCard for testing."""
    return {
        "record_id": "record_001",
        "study_type": "rct",  # Lowercase enum value
        "sample_size": 500,  # Correct field name
        "population_extracted": "Adults 40-70 years with T2DM, HbA1c 7-10%",
        "intervention_extracted": "Semaglutide 1mg weekly for 52 weeks",
        "comparator_extracted": "Placebo injection weekly",
        "primary_outcome": "HbA1c change",  # Single string
        "outcomes": [  # Correct field name
            {
                "name": "HbA1c change",
                "measure_type": "MD",  # OutcomeMeasureType uses MD, not mean_difference
                "value": -1.2,
                "ci_lower": -1.5,
                "ci_upper": -0.9,
                "p_value": 0.001,
            }
        ],
        "follow_up_duration": "52 weeks",  # Correct field name
        "country": "USA, UK, Germany",  # String, not list
        "funding_source": "Novo Nordisk",
        "supporting_snippet_ids": ["snip_001"],  # Correct field name
    }


@pytest.fixture
def sample_rob2_result() -> dict:
    """Sample RoB2 result for testing."""
    return {
        "record_id": "record_001",
        "domains": [
            {
                "domain": "randomization_process",  # Lowercase enum value
                "judgment": "low",  # Lowercase enum value
                "rationale": "Computer-generated randomization with allocation concealment.",
                "supporting_snippet_ids": ["snip_001"],
            },
            {
                "domain": "deviations_from_intended_interventions",
                "judgment": "low",
                "rationale": "Double-blind design maintained throughout.",
                "supporting_snippet_ids": ["snip_001"],
            },
            {
                "domain": "missing_outcome_data",
                "judgment": "some_concerns",
                "rationale": "15% dropout, but ITT analysis performed.",
                "supporting_snippet_ids": ["snip_001"],
            },
            {
                "domain": "measurement_of_outcome",
                "judgment": "low",
                "rationale": "Objective outcome (HbA1c) measured by blinded lab.",
                "supporting_snippet_ids": ["snip_001"],
            },
            {
                "domain": "selection_of_reported_result",
                "judgment": "low",
                "rationale": "Pre-registered protocol, all outcomes reported.",
                "supporting_snippet_ids": ["snip_001"],
            },
        ],
        "overall_judgment": "low",  # Required field
        "overall_rationale": "Low risk of bias across all domains.",  # Required field
    }


@pytest.fixture
def sample_evidence_claim() -> dict:
    """Sample evidence claim for testing."""
    return {
        "claim_id": "claim_001",  # Correct field name
        "claim_text": "GLP-1 receptor agonists reduce HbA1c by approximately 1.2% compared to placebo.",  # Correct field name
        "supporting_snippet_ids": ["snip_001"],
        "certainty": "high",  # Lowercase enum value, correct field name
        "certainty_rationale": "Large RCTs with consistent results, precise estimates.",
    }

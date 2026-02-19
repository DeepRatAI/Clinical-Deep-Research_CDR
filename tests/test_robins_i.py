"""
Tests for ROBINS-I Assessor

HIGH-3: Validates ROBINS-I assessment for observational studies.
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-3
"""

from unittest.mock import MagicMock, patch

import pytest

from cdr.core.enums import ROBINSIDomain, ROBINSIJudgment
from cdr.core.schemas import ROBINSIResult


# --- Unit Tests for ROBINSIAssessor ---


class TestROBINSIAssessorParsing:
    """Test ROBINS-I response parsing logic."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        provider = MagicMock()
        return provider

    def test_parse_valid_robinsi_response(self, mock_provider):
        """Parse well-formed ROBINS-I JSON response."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        # Patch the tracer and metrics
        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            valid_json = """{
                "domains": [
                    {"domain": "confounding", "judgment": "SERIOUS", "rationale": "Inadequate confounder control"},
                    {"domain": "selection", "judgment": "MODERATE", "rationale": "Some selection bias possible"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Well-defined intervention"},
                    {"domain": "deviations", "judgment": "LOW", "rationale": "No major deviations noted"},
                    {"domain": "missing_data", "judgment": "MODERATE", "rationale": "10% missing data"},
                    {"domain": "measurement", "judgment": "LOW", "rationale": "Objective outcomes"},
                    {"domain": "selection_reported", "judgment": "LOW", "rationale": "Pre-registered analysis"}
                ]
            }"""

            result = assessor._parse_response("PMC123", valid_json)

            assert isinstance(result, ROBINSIResult)
            assert result.record_id == "PMC123"
            assert len(result.domains) == 7
            # Overall should be SERIOUS (highest severity)
            assert result.overall_judgment == ROBINSIJudgment.SERIOUS

    def test_parse_response_with_alternative_domain_names(self, mock_provider):
        """Parse response with verbose domain names."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            # LLMs might use verbose domain names
            verbose_json = """{
                "domains": [
                    {"domain": "bias_due_to_confounding", "judgment": "LOW", "rationale": "Good confounder control throughout study"},
                    {"domain": "bias_in_selection_of_participants", "judgment": "LOW", "rationale": "Random selection of participants"},
                    {"domain": "bias_in_classification_of_interventions", "judgment": "LOW", "rationale": "Clear intervention classification"},
                    {"domain": "bias_due_to_deviations_from_intended_interventions", "judgment": "LOW", "rationale": "No deviations from protocol"},
                    {"domain": "bias_due_to_missing_data", "judgment": "LOW", "rationale": "Complete data collection achieved"},
                    {"domain": "bias_in_measurement_of_outcomes", "judgment": "LOW", "rationale": "Blinded outcome assessment"},
                    {"domain": "bias_in_selection_of_reported_result", "judgment": "LOW", "rationale": "Pre-registered analysis plan"}
                ]
            }"""

            result = assessor._parse_response("PMC456", verbose_json)

            assert result.overall_judgment == ROBINSIJudgment.LOW
            assert all(d.judgment == ROBINSIJudgment.LOW for d in result.domains)

    def test_parse_response_adds_missing_domains(self, mock_provider):
        """Missing domains should be added as NO_INFORMATION."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            # Only 3 domains provided
            partial_json = """{
                "domains": [
                    {"domain": "confounding", "judgment": "LOW", "rationale": "Good confounder control"},
                    {"domain": "selection", "judgment": "LOW", "rationale": "Appropriate selection method"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Clear intervention classification"}
                ]
            }"""

            result = assessor._parse_response("PMC789", partial_json)

            assert len(result.domains) == 7  # All 7 domains
            # Overall should be NO_INFORMATION because some domains are missing
            assert result.overall_judgment == ROBINSIJudgment.NO_INFORMATION

    def test_parse_response_handles_markdown_wrapper(self, mock_provider):
        """Handle JSON wrapped in markdown code blocks."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            wrapped_json = """```json
{
    "domains": [
        {"domain": "confounding", "judgment": "CRITICAL", "rationale": "No confounder control at all"},
        {"domain": "selection", "judgment": "CRITICAL", "rationale": "Self-selected participants"},
        {"domain": "classification", "judgment": "CRITICAL", "rationale": "Unclear intervention definition"},
        {"domain": "deviations", "judgment": "CRITICAL", "rationale": "Many protocol deviations observed"},
        {"domain": "missing_data", "judgment": "CRITICAL", "rationale": "50% of data lost to follow-up"},
        {"domain": "measurement", "judgment": "CRITICAL", "rationale": "Self-reported outcomes only"},
        {"domain": "selection_reported", "judgment": "CRITICAL", "rationale": "Post-hoc outcome selection"}
    ]
}
```"""

            result = assessor._parse_response("PMC999", wrapped_json)

            assert result.overall_judgment == ROBINSIJudgment.CRITICAL


class TestROBINSIOverallJudgment:
    """Test overall judgment calculation per ROBINS-I algorithm."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock LLM provider."""
        return MagicMock()

    def test_overall_critical_if_any_critical(self, mock_provider):
        """Overall CRITICAL if any domain is CRITICAL."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            # 6 LOW + 1 CRITICAL = CRITICAL overall
            json_resp = """{
                "domains": [
                    {"domain": "confounding", "judgment": "CRITICAL", "rationale": "Fatal flaw in confounder control"},
                    {"domain": "selection", "judgment": "LOW", "rationale": "Appropriate selection method used"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Intervention well classified"},
                    {"domain": "deviations", "judgment": "LOW", "rationale": "No major protocol deviations"},
                    {"domain": "missing_data", "judgment": "LOW", "rationale": "Complete data available"},
                    {"domain": "measurement", "judgment": "LOW", "rationale": "Valid outcome measures used"},
                    {"domain": "selection_reported", "judgment": "LOW", "rationale": "Pre-specified analysis plan"}
                ]
            }"""

            result = assessor._parse_response("PMC_CRIT", json_resp)
            assert result.overall_judgment == ROBINSIJudgment.CRITICAL
            assert "confounding" in result.overall_rationale.lower()

    def test_overall_serious_if_any_serious(self, mock_provider):
        """Overall SERIOUS if any domain is SERIOUS (no CRITICAL)."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            json_resp = """{
                "domains": [
                    {"domain": "confounding", "judgment": "SERIOUS", "rationale": "Serious problem with confounding"},
                    {"domain": "selection", "judgment": "LOW", "rationale": "Selection criteria appropriate"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Intervention properly classified"},
                    {"domain": "deviations", "judgment": "MODERATE", "rationale": "Minor protocol deviations"},
                    {"domain": "missing_data", "judgment": "LOW", "rationale": "Data completeness verified"},
                    {"domain": "measurement", "judgment": "LOW", "rationale": "Outcome measurement valid"},
                    {"domain": "selection_reported", "judgment": "LOW", "rationale": "Analysis plan pre-registered"}
                ]
            }"""

            result = assessor._parse_response("PMC_SER", json_resp)
            assert result.overall_judgment == ROBINSIJudgment.SERIOUS

    def test_overall_moderate_if_any_moderate(self, mock_provider):
        """Overall MODERATE if any domain is MODERATE (no SERIOUS/CRITICAL)."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            json_resp = """{
                "domains": [
                    {"domain": "confounding", "judgment": "MODERATE", "rationale": "Some residual confounding"},
                    {"domain": "selection", "judgment": "LOW", "rationale": "Selection process adequate"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Intervention clearly defined"},
                    {"domain": "deviations", "judgment": "LOW", "rationale": "Protocol followed correctly"},
                    {"domain": "missing_data", "judgment": "MODERATE", "rationale": "Some missing data present"},
                    {"domain": "measurement", "judgment": "LOW", "rationale": "Measurements standardized"},
                    {"domain": "selection_reported", "judgment": "LOW", "rationale": "Primary outcomes pre-specified"}
                ]
            }"""

            result = assessor._parse_response("PMC_MOD", json_resp)
            assert result.overall_judgment == ROBINSIJudgment.MODERATE

    def test_overall_low_only_if_all_low(self, mock_provider):
        """Overall LOW only if ALL domains are LOW."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            mock_tracer.return_value = MagicMock()
            mock_metrics.return_value = MagicMock()

            assessor = ROBINSIAssessor(provider=mock_provider)

            json_resp = """{
                "domains": [
                    {"domain": "confounding", "judgment": "LOW", "rationale": "Perfect confounder control achieved"},
                    {"domain": "selection", "judgment": "LOW", "rationale": "Randomized selection process"},
                    {"domain": "classification", "judgment": "LOW", "rationale": "Clear intervention definition"},
                    {"domain": "deviations", "judgment": "LOW", "rationale": "No deviations from protocol"},
                    {"domain": "missing_data", "judgment": "LOW", "rationale": "Complete data collection"},
                    {"domain": "measurement", "judgment": "LOW", "rationale": "Blinded outcome assessment"},
                    {"domain": "selection_reported", "judgment": "LOW", "rationale": "Pre-registered analysis"}
                ]
            }"""

            result = assessor._parse_response("PMC_LOW", json_resp)
            assert result.overall_judgment == ROBINSIJudgment.LOW


class TestROBINSIAssessorIntegration:
    """Integration tests for ROBINSIAssessor with mocked LLM."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create mock LLM response."""
        response = MagicMock()
        response.content = """{
            "domains": [
                {"domain": "confounding", "judgment": "MODERATE", "rationale": "Adjusted for age and sex only"},
                {"domain": "selection", "judgment": "LOW", "rationale": "Population-based registry used"},
                {"domain": "classification", "judgment": "LOW", "rationale": "ICD codes used consistently"},
                {"domain": "deviations", "judgment": "LOW", "rationale": "ITT analysis conducted properly"},
                {"domain": "missing_data", "judgment": "MODERATE", "rationale": "15% lost to follow-up noted"},
                {"domain": "measurement", "judgment": "LOW", "rationale": "Lab-confirmed outcome measures"},
                {"domain": "selection_reported", "judgment": "LOW", "rationale": "Protocol registered beforehand"}
            ]
        }"""
        response.total_tokens = 500
        return response

    def test_assess_returns_robins_i_result(self, mock_llm_response):
        """Full assess() call returns ROBINSIResult."""
        from cdr.rob2.robins_i_assessor import ROBINSIAssessor

        mock_provider = MagicMock()
        mock_provider.complete.return_value = mock_llm_response

        with (
            patch("cdr.rob2.robins_i_assessor.get_tracer") as mock_tracer,
            patch("cdr.rob2.robins_i_assessor.get_cdr_metrics") as mock_metrics,
        ):
            # Set up tracer mock to return proper context manager
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            mock_tracer.return_value.span.return_value = mock_span

            mock_metrics_obj = MagicMock()
            mock_metrics_obj.llm_requests = MagicMock()
            mock_metrics_obj.llm_tokens = MagicMock()
            mock_metrics.return_value = mock_metrics_obj

            assessor = ROBINSIAssessor(provider=mock_provider)

            result = assessor.assess(
                record_id="PMC_COHORT",
                text="This cohort study followed 5000 patients over 10 years...",
                study_info={"study_type": "cohort", "population_n": 5000},
            )

            assert isinstance(result, ROBINSIResult)
            assert result.record_id == "PMC_COHORT"
            assert result.overall_judgment == ROBINSIJudgment.MODERATE
            mock_provider.complete.assert_called_once()


class TestROBINSIDomainEnums:
    """Test ROBINS-I domain enums are correct per specification."""

    def test_all_seven_domains_exist(self):
        """ROBINS-I has exactly 7 domains."""
        assert len(ROBINSIDomain) == 7

    def test_domain_names_match_specification(self):
        """Domain names match ROBINS-I specification (verbose format)."""
        # ROBINSIDomain uses verbose domain names per ROBINS-I documentation
        expected_domains = {
            "bias_due_to_confounding",
            "bias_in_selection_of_participants",
            "bias_in_classification_of_interventions",
            "bias_due_to_deviations_from_intended_interventions",
            "bias_due_to_missing_data",
            "bias_in_measurement_of_outcomes",
            "bias_in_selection_of_reported_result",
        }

        actual_domains = {d.value for d in ROBINSIDomain}
        assert actual_domains == expected_domains

    def test_judgment_levels_correct(self):
        """ROBINS-I uses 5-level judgment scale."""
        assert len(ROBINSIJudgment) == 5

        expected_judgments = {"low", "moderate", "serious", "critical", "no_information"}
        actual_judgments = {j.value for j in ROBINSIJudgment}
        assert actual_judgments == expected_judgments


class TestROBINSIVsRoB2Routing:
    """Test that routing correctly separates RCT vs observational."""

    def test_cohort_uses_robins_i(self):
        """Cohort studies should use ROBINS-I, not RoB2."""
        from cdr.core.enums import StudyType

        robins_i_study_types = {
            StudyType.COHORT,
            StudyType.CASE_CONTROL,
            StudyType.CROSS_SECTIONAL,
        }

        assert StudyType.COHORT in robins_i_study_types
        assert StudyType.RCT not in robins_i_study_types

    def test_rct_uses_rob2(self):
        """RCT studies should use RoB2, not ROBINS-I."""
        from cdr.core.enums import StudyType

        rob2_study_types = {
            StudyType.RCT,
            StudyType.META_ANALYSIS,
            StudyType.SYSTEMATIC_REVIEW,
        }

        assert StudyType.RCT in rob2_study_types
        assert StudyType.COHORT not in rob2_study_types

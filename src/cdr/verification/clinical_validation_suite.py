"""
Clinical Validation Suite for DoD3 Compliance

This module provides a comprehensive validation suite with:
- 3 POSITIVE controls (should be publishable)
- 3 NEGATIVE controls (should be unpublishable)
- 3 BORDERLINE cases (partial publishability)

Based on real clinical trial patterns from registries:
- ClinicalTrials.gov
- ISRCTN
- ANZCTR

DoD3 Contract Requirements:
- Type-checking + gates determinísticos + trazabilidad completa
- Controles positivos = publicación correcta auditable
- Controles negativos = rechazo correcto auditable

Author: CDR System
Date: 2026-02-01
Refs: CDR_Agent_Guidance_and_Development_Protocol.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cdr.core.schemas import PICO, Record, Snippet, SourceRef, EvidenceClaim
from cdr.core.enums import (
    StudyType,
    RecordSource,
    GRADECertainty,
    Section,
    ComparatorSource,
)
from cdr.verification.dod3_gates import DoD3Validator


# =============================================================================
# TEST CASE DEFINITION
# =============================================================================


@dataclass
class ValidationCase:
    """A single validation case for clinical auditing."""

    case_id: str
    name: str
    description: str
    category: str  # "positive", "negative", "borderline"
    pico: PICO
    records: list[Record]
    snippets: list[Snippet]
    claims: list[EvidenceClaim]

    # Expectations
    expected_status: str  # "publishable", "unpublishable", "partially_publishable"
    expected_blockers: int = 0  # Number of expected blocker violations
    expected_gates_pass: list[str] = field(default_factory=list)
    expected_gates_fail: list[str] = field(default_factory=list)

    # Registry source info
    registry_source: str | None = None  # e.g., "ClinicalTrials.gov NCT01234567"
    real_trial_pattern: str | None = None  # Pattern this case is based on

    # Tags for filtering
    tags: list[str] = field(default_factory=list)


def create_record(
    record_id: str,
    title: str,
    abstract: str,
    pmid: str | None = None,
    publication_type: list[str] | None = None,
    source: RecordSource = RecordSource.PUBMED,
) -> Record:
    """Helper to create test records."""
    return Record(
        record_id=record_id,
        source=source,
        content_hash=f"hash_{record_id}",
        title=title,
        abstract=abstract,
        pmid=pmid,
        publication_type=publication_type or [],
    )


def create_snippet(
    snippet_id: str,
    text: str,
    record_id: str,
    pmid: str | None = None,
    role: str = "result",
) -> Snippet:
    """Helper to create test snippets with role annotation."""
    snippet = Snippet(
        snippet_id=snippet_id,
        text=text,
        source_ref=SourceRef(
            record_id=record_id,
            pmid=pmid,
        ),
        section=Section.RESULTS if role == "result" else Section.ABSTRACT,
    )
    # Add role as custom attribute
    snippet._role = role
    return snippet


def create_claim(
    claim_id: str,
    claim_text: str,
    supporting_snippet_ids: list[str],
    certainty: GRADECertainty = GRADECertainty.LOW,
    grade_rationale: dict | None = None,
) -> EvidenceClaim:
    """Helper to create test claims with GRADE rationale."""
    return EvidenceClaim(
        claim_id=claim_id,
        claim_text=claim_text,
        certainty=certainty,
        supporting_snippet_ids=supporting_snippet_ids,
        grade_rationale=grade_rationale,
    )


# =============================================================================
# POSITIVE CONTROLS (PUBLISHABLE)
# =============================================================================


def get_positive_controls() -> list[ValidationCase]:
    """
    Get 3 POSITIVE control cases that MUST be publishable.

    These represent clean RCTs with:
    - Explicit population matching
    - Clear, user-specified comparator
    - Correct study type (RCT)
    - No context purity violations
    - Strong evidence support
    """
    cases = []

    # -------------------------------------------------------------------------
    # POSITIVE 1: Clean RCT - Metformin vs Placebo for Glycemic Control
    # Pattern: NCT00766857 (DPP study pattern)
    # -------------------------------------------------------------------------
    pico_metformin = PICO(
        population="adults with prediabetes",
        intervention="metformin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="progression to type 2 diabetes",
        study_types=[StudyType.RCT],
    )

    record_metformin = create_record(
        record_id="pos_001",
        title="Metformin for Prevention of Type 2 Diabetes: A Randomized Controlled Trial",
        abstract=(
            "Background: Adults with prediabetes are at high risk for type 2 diabetes. "
            "Methods: We randomized 3234 adults with prediabetes to receive metformin 850mg "
            "twice daily or placebo. Primary outcome was progression to type 2 diabetes. "
            "Mean follow-up was 2.8 years. "
            "Results: Metformin reduced the incidence of type 2 diabetes by 31% compared to placebo "
            "(HR 0.69, 95% CI 0.57-0.83, p<0.001) in adults with prediabetes. "
            "Conclusion: Metformin is effective for diabetes prevention in prediabetic adults."
        ),
        pmid="11832527",
        publication_type=["Randomized Controlled Trial", "Multicenter Study"],
    )

    snippet_metformin = create_snippet(
        snippet_id="pos_001_snip_1",
        text=(
            "In our randomized trial of adults with prediabetes, metformin 850mg twice daily "
            "reduced the incidence of type 2 diabetes by 31% compared to placebo "
            "(HR 0.69, 95% CI 0.57-0.83, p<0.001). This effect was consistent across subgroups "
            "defined by age, sex, and baseline glucose."
        ),
        record_id="pos_001",
        pmid="11832527",
        role="result",
    )

    claim_metformin = create_claim(
        claim_id="pos_001_claim_1",
        claim_text=(
            "Metformin reduces the incidence of type 2 diabetes in adults with prediabetes "
            "compared to placebo (RR reduction 31%)."
        ),
        supporting_snippet_ids=["pos_001_snip_1"],
        certainty=GRADECertainty.HIGH,
        grade_rationale={
            "risk_of_bias": "low - double-blind RCT with allocation concealment",
            "inconsistency": "low - consistent effect across subgroups",
            "indirectness": "low - direct population and intervention match",
            "imprecision": "low - narrow confidence interval",
            "publication_bias": "low - large trial with preregistration",
        },
    )

    cases.append(
        ValidationCase(
            case_id="POS-001",
            name="Metformin vs Placebo in Prediabetes",
            description="Clean RCT with explicit comparator, matching population, strong evidence",
            category="positive",
            pico=pico_metformin,
            records=[record_metformin],
            snippets=[snippet_metformin],
            claims=[claim_metformin],
            expected_status="publishable",
            expected_blockers=0,
            expected_gates_pass=[
                "pico_match",
                "study_type",
                "context_purity",
                "assertion_coverage",
            ],
            registry_source="ClinicalTrials.gov (DPP pattern)",
            real_trial_pattern="DPP - Diabetes Prevention Program",
            tags=["positive", "clean_rct", "diabetes", "baseline"],
        )
    )

    # -------------------------------------------------------------------------
    # POSITIVE 2: Clean RCT - Statins vs Placebo for Cardiovascular Prevention
    # Pattern: JUPITER trial (NCT00239681)
    # -------------------------------------------------------------------------
    pico_statin = PICO(
        population="adults with elevated CRP and low LDL cholesterol",
        intervention="rosuvastatin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="major cardiovascular events",
        study_types=[StudyType.RCT],
    )

    record_statin = create_record(
        record_id="pos_002",
        title="Rosuvastatin for Primary Prevention in Adults with High CRP: JUPITER Trial",
        abstract=(
            "Background: Adults with elevated high-sensitivity C-reactive protein (hsCRP) "
            "and normal LDL cholesterol may benefit from statin therapy. "
            "Methods: We randomized 17,802 adults with elevated CRP (≥2 mg/L) and LDL <130 mg/dL "
            "to rosuvastatin 20mg daily or placebo. Primary endpoint was major cardiovascular events. "
            "Results: Rosuvastatin reduced the primary endpoint by 44% compared to placebo "
            "(HR 0.56, 95% CI 0.46-0.69, p<0.00001) in adults with elevated CRP and low LDL. "
            "Conclusion: Rosuvastatin prevents cardiovascular events in this high-risk population."
        ),
        pmid="18997196",
        publication_type=["Randomized Controlled Trial", "Multicenter Study"],
    )

    snippet_statin = create_snippet(
        snippet_id="pos_002_snip_1",
        text=(
            "Among adults with elevated CRP (≥2 mg/L) and LDL cholesterol <130 mg/dL, "
            "rosuvastatin 20mg daily reduced major cardiovascular events by 44% compared to placebo "
            "(HR 0.56, 95% CI 0.46-0.69, p<0.00001). The number needed to treat was 25 for 5 years."
        ),
        record_id="pos_002",
        pmid="18997196",
        role="result",
    )

    claim_statin = create_claim(
        claim_id="pos_002_claim_1",
        claim_text=(
            "Rosuvastatin reduces major cardiovascular events in adults with elevated CRP "
            "and normal LDL compared to placebo."
        ),
        supporting_snippet_ids=["pos_002_snip_1"],
        certainty=GRADECertainty.HIGH,
        grade_rationale={
            "risk_of_bias": "low - industry-funded but rigorous design",
            "inconsistency": "low - consistent across prespecified subgroups",
            "indirectness": "low - population and intervention directly applicable",
            "imprecision": "low - highly significant with narrow CI",
            "publication_bias": "low - preregistered, stopped early for benefit",
        },
    )

    cases.append(
        ValidationCase(
            case_id="POS-002",
            name="Rosuvastatin vs Placebo in High-CRP Adults",
            description="JUPITER pattern - clean primary prevention RCT",
            category="positive",
            pico=pico_statin,
            records=[record_statin],
            snippets=[snippet_statin],
            claims=[claim_statin],
            expected_status="publishable",
            expected_blockers=0,
            expected_gates_pass=["pico_match", "study_type", "context_purity"],
            registry_source="ClinicalTrials.gov NCT00239681",
            real_trial_pattern="JUPITER Trial",
            tags=["positive", "clean_rct", "cardiovascular", "statin"],
        )
    )

    # -------------------------------------------------------------------------
    # POSITIVE 3: Clean RCT - ACE Inhibitor vs Placebo for Heart Failure
    # Pattern: CONSENSUS trial (clean HF RCT)
    # -------------------------------------------------------------------------
    pico_acei = PICO(
        population="patients with severe heart failure",
        intervention="enalapril",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="mortality",
        study_types=[StudyType.RCT],
    )

    record_acei = create_record(
        record_id="pos_003",
        title="Enalapril for Severe Heart Failure: The CONSENSUS Trial",
        abstract=(
            "Background: Patients with severe heart failure (NYHA IV) have high mortality. "
            "Methods: We randomized 253 patients with severe heart failure to enalapril "
            "or placebo, added to conventional therapy. Primary outcome was mortality. "
            "Results: Enalapril reduced mortality by 40% at 6 months compared to placebo "
            "(HR 0.60, 95% CI 0.38-0.95, p=0.027) in patients with severe heart failure. "
            "Conclusion: Enalapril reduces mortality in severe heart failure."
        ),
        pmid="2883575",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_acei = create_snippet(
        snippet_id="pos_003_snip_1",
        text=(
            "In patients with severe heart failure (NYHA class IV), enalapril reduced "
            "6-month mortality by 40% compared to placebo (HR 0.60, 95% CI 0.38-0.95, p=0.027). "
            "The trial was stopped early due to clear mortality benefit."
        ),
        record_id="pos_003",
        pmid="2883575",
        role="result",
    )

    claim_acei = create_claim(
        claim_id="pos_003_claim_1",
        claim_text=(
            "Enalapril reduces mortality in patients with severe heart failure compared to placebo."
        ),
        supporting_snippet_ids=["pos_003_snip_1"],
        certainty=GRADECertainty.HIGH,
        grade_rationale={
            "risk_of_bias": "low - double-blind RCT",
            "inconsistency": "low - landmark trial with consistent replication",
            "indirectness": "low - direct match to PICO",
            "imprecision": "low - significant mortality benefit",
            "publication_bias": "low - foundational study",
        },
    )

    cases.append(
        ValidationCase(
            case_id="POS-003",
            name="Enalapril vs Placebo in Severe Heart Failure",
            description="CONSENSUS pattern - classic mortality RCT",
            category="positive",
            pico=pico_acei,
            records=[record_acei],
            snippets=[snippet_acei],
            claims=[claim_acei],
            expected_status="publishable",
            expected_blockers=0,
            expected_gates_pass=["pico_match", "study_type", "context_purity"],
            registry_source="CONSENSUS Trial (Sweden)",
            real_trial_pattern="CONSENSUS I",
            tags=["positive", "clean_rct", "heart_failure", "mortality"],
        )
    )

    return cases


# =============================================================================
# NEGATIVE CONTROLS (UNPUBLISHABLE)
# =============================================================================


def get_negative_controls() -> list[ValidationCase]:
    """
    Get 3 NEGATIVE control cases that MUST be unpublishable.

    These represent clear DoD3 violations:
    - Population exclusion (ASPREE pattern)
    - Comparator mismatch (head-to-head when asking placebo)
    - Study type mismatch (cohort when RCT required)
    """
    cases = []

    # -------------------------------------------------------------------------
    # NEGATIVE 1: Population EXCLUDED (ASPREE pattern)
    # The evidence explicitly excludes the target population
    # -------------------------------------------------------------------------
    pico_af_aspirin = PICO(
        population="patients with atrial fibrillation",
        intervention="aspirin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )

    record_aspree = create_record(
        record_id="neg_001",
        title="ASPREE: Aspirin in Healthy Elderly Without Cardiovascular Disease",
        abstract=(
            "Background: The role of aspirin in primary prevention is debated. "
            "Methods: We randomized 19,114 healthy elderly adults WITHOUT cardiovascular disease, "
            "atrial fibrillation, or dementia to aspirin 100mg daily or placebo. "
            "Exclusion criteria included: history of atrial fibrillation, prior CVD, diabetes. "
            "Primary outcome was disability-free survival. "
            "Results: Aspirin did not reduce cardiovascular events in this healthy elderly population "
            "without atrial fibrillation (HR 0.95, 95% CI 0.83-1.08). "
            "Conclusion: Aspirin provides no net benefit in healthy elderly without CVD or AF."
        ),
        pmid="30221595",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_aspree = create_snippet(
        snippet_id="neg_001_snip_1",
        text=(
            "We enrolled healthy elderly adults WITHOUT atrial fibrillation, cardiovascular disease, "
            "or dementia. Patients with atrial fibrillation were explicitly excluded. "
            "Aspirin 100mg daily did not reduce cardiovascular events compared to placebo "
            "in this population without atrial fibrillation."
        ),
        record_id="neg_001",
        pmid="30221595",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="NEG-001",
            name="ASPREE - Population WITHOUT AF (Excluded)",
            description="BLOCKER: Evidence explicitly excludes atrial fibrillation patients",
            category="negative",
            pico=pico_af_aspirin,
            records=[record_aspree],
            snippets=[snippet_aspree],
            claims=[],
            expected_status="unpublishable",
            expected_blockers=1,
            expected_gates_fail=["pico_match"],
            registry_source="ANZCTR ACTRN12607000549482",
            real_trial_pattern="ASPREE Trial",
            tags=["negative", "population_excluded", "aspree", "blocker"],
        )
    )

    # -------------------------------------------------------------------------
    # NEGATIVE 2: Comparator MISMATCH (DOAC vs Aspirin when asking Placebo)
    # User asks about aspirin vs placebo, evidence is DOAC vs aspirin
    # -------------------------------------------------------------------------
    record_aristotle = create_record(
        record_id="neg_002",
        title="Apixaban versus Aspirin in Patients with Atrial Fibrillation",
        abstract=(
            "Background: Patients with atrial fibrillation need stroke prevention. "
            "Methods: We randomized 5599 patients with atrial fibrillation to apixaban 5mg "
            "twice daily or aspirin. No placebo arm was included. "
            "Results: Apixaban reduced stroke by 55% compared to aspirin "
            "(HR 0.45, 95% CI 0.32-0.62) in patients with atrial fibrillation. "
            "Conclusion: Apixaban is superior to aspirin for stroke prevention in AF."
        ),
        pmid="21309657",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_aristotle = create_snippet(
        snippet_id="neg_002_snip_1",
        text=(
            "In patients with atrial fibrillation unsuitable for warfarin, "
            "apixaban 5mg twice daily reduced stroke risk by 55% compared to aspirin "
            "(HR 0.45, 95% CI 0.32-0.62). Apixaban is superior to aspirin, not placebo."
        ),
        record_id="neg_002",
        pmid="21309657",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="NEG-002",
            name="AVERROES - DOAC vs Aspirin (Wrong Comparator)",
            description="BLOCKER: Evidence compares apixaban to aspirin, not aspirin to placebo",
            category="negative",
            pico=pico_af_aspirin,
            records=[record_aristotle],
            snippets=[snippet_aristotle],
            claims=[],
            expected_status="unpublishable",
            expected_blockers=1,
            expected_gates_fail=["comparator_alignment"],
            registry_source="ClinicalTrials.gov NCT00496769",
            real_trial_pattern="AVERROES Trial",
            tags=["negative", "comparator_mismatch", "blocker"],
        )
    )

    # -------------------------------------------------------------------------
    # NEGATIVE 3: Study Type MISMATCH (Cohort when RCT required)
    # -------------------------------------------------------------------------
    record_cohort = create_record(
        record_id="neg_003",
        title="Aspirin Use and Stroke in Atrial Fibrillation: A Retrospective Cohort",
        abstract=(
            "Background: Limited RCT data exist for aspirin monotherapy in AF. "
            "Methods: This retrospective cohort study analyzed 45,000 AF patients from claims data. "
            "We compared aspirin users to non-users (not placebo). "
            "No randomization was performed. Selection bias is a concern. "
            "Results: Aspirin was associated with 15% lower stroke risk (observational). "
            "Conclusion: Observational data suggest aspirin may reduce stroke in AF."
        ),
        pmid="99887766",
        publication_type=["Observational Study", "Cohort Study", "Retrospective Study"],
    )

    snippet_cohort = create_snippet(
        snippet_id="neg_003_snip_1",
        text=(
            "In this retrospective cohort of 45,000 AF patients, aspirin users had 15% lower "
            "stroke risk than non-users (HR 0.85, 95% CI 0.75-0.95). However, this was an "
            "observational study with potential confounding and selection bias."
        ),
        record_id="neg_003",
        pmid="99887766",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="NEG-003",
            name="Cohort Study When RCT Required",
            description="BLOCKER: PICO requires RCT, evidence is retrospective cohort",
            category="negative",
            pico=pico_af_aspirin,
            records=[record_cohort],
            snippets=[snippet_cohort],
            claims=[],
            expected_status="unpublishable",
            expected_blockers=1,
            expected_gates_fail=["study_type"],
            registry_source="Synthetic (claims-based cohort pattern)",
            real_trial_pattern="Registry cohort study pattern",
            tags=["negative", "study_type_mismatch", "cohort", "blocker"],
        )
    )

    return cases


# =============================================================================
# BORDERLINE CASES (PARTIALLY PUBLISHABLE)
# =============================================================================


def get_borderline_cases() -> list[ValidationCase]:
    """
    Get 3 BORDERLINE cases that test edge conditions.

    These represent:
    - Valid Sub-PICO with original PICO lacking direct evidence
    - Mixed evidence quality requiring stratification
    - Context-specific findings needing decomposition
    """
    cases = []

    # -------------------------------------------------------------------------
    # BORDERLINE 1: Sub-PICO Valid, Original PICO Not Directly Answered
    # User asks aspirin vs placebo in AF, evidence is DOAC vs aspirin in AF
    # Sub-PICO (DOAC vs aspirin) is valid, but doesn't answer original question
    # -------------------------------------------------------------------------
    pico_aspirin_placebo = PICO(
        population="patients with atrial fibrillation",
        intervention="aspirin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )

    # Evidence for Sub-PICO: DOAC vs aspirin
    record_subpico = create_record(
        record_id="border_001",
        title="DOACs versus Aspirin in Atrial Fibrillation: Meta-analysis",
        abstract=(
            "Background: DOACs are compared to aspirin in AF patients. "
            "Methods: We conducted a meta-analysis of RCTs comparing DOACs to aspirin "
            "in patients with atrial fibrillation. "
            "Results: DOACs reduced stroke by 50% compared to aspirin (RR 0.50, 95% CI 0.40-0.65). "
            "No trials directly compared aspirin to placebo in the modern era. "
            "Conclusion: DOACs are superior to aspirin, but aspirin vs placebo data lacking."
        ),
        pmid="55443322",
        publication_type=["Meta-Analysis", "Systematic Review"],
    )

    snippet_subpico = create_snippet(
        snippet_id="border_001_snip_1",
        text=(
            "Meta-analysis of 3 RCTs (n=15,000) showed DOACs reduced stroke by 50% vs aspirin "
            "(RR 0.50, 95% CI 0.40-0.65). No modern RCT data exists for aspirin vs placebo in AF."
        ),
        record_id="border_001",
        pmid="55443322",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="BORDER-001",
            name="Sub-PICO Valid, Original Not Directly Answered",
            description=(
                "DOAC vs aspirin evidence exists (Sub-PICO valid), "
                "but aspirin vs placebo (original) not directly answered"
            ),
            category="borderline",
            pico=pico_aspirin_placebo,
            records=[record_subpico],
            snippets=[snippet_subpico],
            claims=[],
            expected_status="partially_publishable",  # Or unpublishable with Sub-PICO section
            expected_blockers=1,  # Comparator mismatch for original PICO
            expected_gates_fail=["comparator_alignment"],
            registry_source="Synthetic meta-analysis pattern",
            real_trial_pattern="Indirect comparison scenario",
            tags=["borderline", "subpico", "indirect_evidence"],
        )
    )

    # -------------------------------------------------------------------------
    # BORDERLINE 2: Mixed Study Types Requiring Stratification
    # One valid RCT + one cohort study - needs stratification, not rejection
    # -------------------------------------------------------------------------
    pico_diabetes_exercise = PICO(
        population="adults with type 2 diabetes",
        intervention="exercise program",
        comparator="usual care",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="glycemic control",
        study_types=[StudyType.RCT],  # User wants RCT only
    )

    record_rct = create_record(
        record_id="border_002a",
        title="Exercise Program for Diabetes: A Randomized Trial",
        abstract=(
            "Background: Exercise may improve glycemic control in type 2 diabetes. "
            "Methods: We randomized 150 adults with type 2 diabetes to a structured exercise "
            "program or usual care for 12 weeks. "
            "Results: Exercise reduced HbA1c by 0.5% compared to usual care (p<0.01). "
            "Conclusion: Exercise improves glycemic control in type 2 diabetes."
        ),
        pmid="11112233",
        publication_type=["Randomized Controlled Trial"],
    )

    record_cohort_mixed = create_record(
        record_id="border_002b",
        title="Physical Activity and Glycemic Control: A Prospective Cohort",
        abstract=(
            "Background: Observational data on exercise in diabetes is valuable. "
            "Methods: We followed 5000 adults with type 2 diabetes for 5 years. "
            "Physical activity was self-reported. "
            "Results: Higher activity levels associated with lower HbA1c (observational). "
            "Conclusion: Prospective cohort supports exercise benefits."
        ),
        pmid="44556677",
        publication_type=["Cohort Study", "Prospective Study"],
    )

    snippet_rct = create_snippet(
        snippet_id="border_002a_snip_1",
        text=(
            "In adults with type 2 diabetes, a 12-week structured exercise program reduced HbA1c "
            "by 0.5% compared to usual care (95% CI -0.7 to -0.3, p<0.01)."
        ),
        record_id="border_002a",
        pmid="11112233",
        role="result",
    )

    snippet_cohort = create_snippet(
        snippet_id="border_002b_snip_1",
        text=(
            "Over 5 years, adults with type 2 diabetes in the highest physical activity quartile "
            "had 0.3% lower HbA1c than the lowest quartile (observational association)."
        ),
        record_id="border_002b",
        pmid="44556677",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="BORDER-002",
            name="Mixed RCT + Cohort Evidence",
            description=(
                "Valid RCT exists but cohort study also retrieved. "
                "Should stratify, not mix or reject entirely."
            ),
            category="borderline",
            pico=pico_diabetes_exercise,
            records=[record_rct, record_cohort_mixed],
            snippets=[snippet_rct, snippet_cohort],
            claims=[],
            expected_status="partially_publishable",  # RCT portion publishable
            expected_blockers=1,  # Cohort study should be flagged
            expected_gates_fail=["study_type"],  # For the cohort
            expected_gates_pass=["pico_match"],  # For the RCT
            registry_source="Synthetic pattern",
            real_trial_pattern="Mixed evidence scenario",
            tags=["borderline", "mixed_evidence", "stratification"],
        )
    )

    # -------------------------------------------------------------------------
    # BORDERLINE 3: Population Context Mismatch (Subclinical vs Clinical)
    # Evidence from subclinical AF applied to clinical AF question
    # -------------------------------------------------------------------------
    pico_clinical_af = PICO(
        population="patients with clinical atrial fibrillation",
        intervention="anticoagulation",
        comparator="aspirin",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )

    record_subclinical = create_record(
        record_id="border_003",
        title="ARTESiA: Anticoagulation in Subclinical Atrial Fibrillation",
        abstract=(
            "Background: Device-detected subclinical AF (SCAF) is increasingly recognized. "
            "Methods: We randomized patients with subclinical AF detected by implanted devices "
            "to apixaban or aspirin. Clinical AF was excluded. "
            "Results: Apixaban reduced stroke compared to aspirin in subclinical AF "
            "(HR 0.63, 95% CI 0.45-0.88). "
            "Conclusion: Anticoagulation benefits subclinical AF, which differs from clinical AF."
        ),
        pmid="33445566",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_subclinical = create_snippet(
        snippet_id="border_003_snip_1",
        text=(
            "In patients with device-detected subclinical atrial fibrillation (not clinical AF), "
            "apixaban reduced stroke by 37% compared to aspirin (HR 0.63, 95% CI 0.45-0.88). "
            "These findings apply to subclinical AF, not necessarily to clinical AF."
        ),
        record_id="border_003",
        pmid="33445566",
        role="result",
    )

    cases.append(
        ValidationCase(
            case_id="BORDER-003",
            name="Subclinical AF Evidence for Clinical AF Question",
            description=(
                "Evidence from subclinical AF trial, question asks about clinical AF. "
                "Population context mismatch - needs explicit acknowledgment."
            ),
            category="borderline",
            pico=pico_clinical_af,
            records=[record_subclinical],
            snippets=[snippet_subclinical],
            claims=[],
            expected_status="partially_publishable",  # With context caveat
            expected_blockers=1,
            expected_gates_fail=["context_purity"],
            registry_source="ClinicalTrials.gov (ARTESiA pattern)",
            real_trial_pattern="ARTESiA Trial",
            tags=["borderline", "population_context", "subclinical_vs_clinical"],
        )
    )

    return cases


# =============================================================================
# FULL VALIDATION SUITE
# =============================================================================


def get_validation_suite() -> list[ValidationCase]:
    """Get the complete validation suite: 3 positive + 3 negative + 3 borderline."""
    cases = []
    cases.extend(get_positive_controls())
    cases.extend(get_negative_controls())
    cases.extend(get_borderline_cases())
    return cases


# =============================================================================
# HARNESS RUNNER
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a single validation case."""

    case_id: str
    case_name: str
    category: str
    passed: bool
    actual_status: str
    expected_status: str
    actual_blockers: int
    expected_blockers: int
    violations: list[dict]
    message: str

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "category": self.category,
            "passed": self.passed,
            "actual_status": self.actual_status,
            "expected_status": self.expected_status,
            "actual_blockers": self.actual_blockers,
            "expected_blockers": self.expected_blockers,
            "violations": self.violations,
            "message": self.message,
        }


@dataclass
class SuiteResult:
    """Overall validation suite result."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    positive_passed: int
    positive_total: int
    negative_passed: int
    negative_total: int
    borderline_passed: int
    borderline_total: int
    results: list[ValidationResult]

    @property
    def success(self) -> bool:
        # All positive controls must pass for publishable
        # All negative controls must pass for unpublishable
        return (
            self.positive_passed == self.positive_total
            and self.negative_passed == self.negative_total
        )

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "positive": {"passed": self.positive_passed, "total": self.positive_total},
            "negative": {"passed": self.negative_passed, "total": self.negative_total},
            "borderline": {"passed": self.borderline_passed, "total": self.borderline_total},
            "results": [r.to_dict() for r in self.results],
        }


def run_validation_suite(
    cases: list[ValidationCase] | None = None, verbose: bool = True
) -> SuiteResult:
    """Run the clinical validation suite."""

    if cases is None:
        cases = get_validation_suite()

    validator = DoD3Validator(strict=True)
    results = []

    counters = {
        "positive": {"passed": 0, "total": 0},
        "negative": {"passed": 0, "total": 0},
        "borderline": {"passed": 0, "total": 0},
    }

    for case in cases:
        counters[case.category]["total"] += 1

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{case.case_id}] {case.name}")
            print(f"Category: {case.category.upper()}")
            print(f"Expected: {case.expected_status}")

        try:
            # Run validation
            validation = validator.validate(
                run_id=f"validation_{case.case_id}",
                pico=case.pico,
                records=case.records,
                snippets=case.snippets,
                claims=case.claims,
            )

            # Determine actual status
            if validation.passed:
                actual_status = "publishable"
            else:
                actual_status = "unpublishable"

            actual_blockers = len(validation.gate_report.blocker_violations)
            violations = [v.to_dict() for v in validation.gate_report.blocker_violations]

            # Determine if test passed
            if case.category == "positive":
                # Positive controls must be publishable
                test_passed = actual_status == "publishable"
            elif case.category == "negative":
                # Negative controls must be unpublishable
                test_passed = actual_status == "unpublishable"
            else:
                # Borderline: check for expected behavior
                test_passed = actual_blockers == case.expected_blockers

            if test_passed:
                counters[case.category]["passed"] += 1

            message = ""
            if not test_passed:
                message = f"Expected {case.expected_status}, got {actual_status} ({actual_blockers} blockers)"

            result = ValidationResult(
                case_id=case.case_id,
                case_name=case.name,
                category=case.category,
                passed=test_passed,
                actual_status=actual_status,
                expected_status=case.expected_status,
                actual_blockers=actual_blockers,
                expected_blockers=case.expected_blockers,
                violations=violations,
                message=message,
            )

            if verbose:
                status_icon = "✅ PASS" if test_passed else "❌ FAIL"
                print(f"Result: {status_icon}")
                print(f"Actual: {actual_status} ({actual_blockers} blockers)")
                if message:
                    print(f"Message: {message}")

        except Exception as e:
            result = ValidationResult(
                case_id=case.case_id,
                case_name=case.name,
                category=case.category,
                passed=False,
                actual_status="error",
                expected_status=case.expected_status,
                actual_blockers=0,
                expected_blockers=case.expected_blockers,
                violations=[],
                message=f"Exception: {str(e)}",
            )
            if verbose:
                print(f"Result: ❌ ERROR - {e}")

        results.append(result)

    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    suite_result = SuiteResult(
        total_cases=len(cases),
        passed_cases=passed_count,
        failed_cases=failed_count,
        positive_passed=counters["positive"]["passed"],
        positive_total=counters["positive"]["total"],
        negative_passed=counters["negative"]["passed"],
        negative_total=counters["negative"]["total"],
        borderline_passed=counters["borderline"]["passed"],
        borderline_total=counters["borderline"]["total"],
        results=results,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("VALIDATION SUITE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total: {suite_result.passed_cases}/{suite_result.total_cases} passed")
        print(
            f"  Positive (publishable): {suite_result.positive_passed}/{suite_result.positive_total}"
        )
        print(
            f"  Negative (unpublishable): {suite_result.negative_passed}/{suite_result.negative_total}"
        )
        print(f"  Borderline: {suite_result.borderline_passed}/{suite_result.borderline_total}")
        print(f"Overall: {'✅ SUCCESS' if suite_result.success else '❌ FAILURE'}")

    return suite_result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Main entry point for validation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="CDR Clinical Validation Suite")
    parser.add_argument("--output", "-o", type=str, help="Path to save results JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        choices=["positive", "negative", "borderline", "all"],
        default="all",
        help="Run only specific category",
    )

    args = parser.parse_args()

    # Get cases
    if args.category == "all":
        cases = get_validation_suite()
    elif args.category == "positive":
        cases = get_positive_controls()
    elif args.category == "negative":
        cases = get_negative_controls()
    else:
        cases = get_borderline_cases()

    # Run suite
    result = run_validation_suite(cases, verbose=not args.quiet)

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()

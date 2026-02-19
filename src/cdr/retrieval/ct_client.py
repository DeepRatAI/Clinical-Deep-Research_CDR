"""
ClinicalTrials.gov Client

Client for ClinicalTrials.gov API v2 to search clinical trial registrations.

Documentation: https://clinicaltrials.gov/data-api/api

Key API Parameters:
- query.term: Free-text search (keep short, <500 chars)
- filter.overallStatus: RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING, etc.
- filter.studyType: INTERVENTIONAL, OBSERVATIONAL, EXPANDED_ACCESS
- filter.phase: PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1, NA
- pageSize: Results per page (max 1000)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from cdr.config import get_settings
from cdr.core.enums import RecordSource, StudyType
from cdr.core.exceptions import ClinicalTrialsError
from cdr.core.schemas import Record
from cdr.observability import get_tracer


CT_API_BASE = "https://clinicaltrials.gov/api/v2"


# Valid filter values according to ClinicalTrials.gov Data API
CT_STUDY_STATUSES = [
    "RECRUITING",
    "COMPLETED",
    "ACTIVE_NOT_RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "TERMINATED",
    "SUSPENDED",
    "WITHDRAWN",
    "AVAILABLE",
    "NO_LONGER_AVAILABLE",
    "TEMPORARILY_NOT_AVAILABLE",
    "APPROVED_FOR_MARKETING",
    "WITHHELD",
    "UNKNOWN",
]

CT_STUDY_TYPES = [
    "INTERVENTIONAL",
    "OBSERVATIONAL",
    "EXPANDED_ACCESS",
]

CT_PHASES = [
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE2",
    "PHASE3",
    "PHASE4",
    "NA",  # Not Applicable
]


@dataclass
class CTSearchFilters:
    """Structured filters for ClinicalTrials.gov search.

    Documentation: https://clinicaltrials.gov/data-api/api

    Note: The API uses 'aggFilters' parameter, not 'filter.*' parameters.
    Format: aggFilters=studyType:int (for INTERVENTIONAL)
    """

    study_statuses: list[str] = field(default_factory=list)
    study_type: str | None = None
    phases: list[str] = field(default_factory=list)

    def to_params(self) -> dict[str, str]:
        """Convert filters to API parameters.

        Uses aggFilters parameter which is the correct format for CT.gov v2 API.
        """
        params = {}
        agg_filters = []

        if self.study_type:
            # Map study type to aggFilters format
            type_map = {
                "INTERVENTIONAL": "int",
                "OBSERVATIONAL": "obs",
                "EXPANDED_ACCESS": "exp",
            }
            if self.study_type in type_map:
                agg_filters.append(f"studyType:{type_map[self.study_type]}")

        if self.phases:
            # Map phases to aggFilters format
            phase_map = {
                "EARLY_PHASE1": "early1",
                "PHASE1": "1",
                "PHASE2": "2",
                "PHASE3": "3",
                "PHASE4": "4",
                "NA": "na",
            }
            for phase in self.phases:
                if phase in phase_map:
                    agg_filters.append(f"phase:{phase_map[phase]}")

        if self.study_statuses:
            # Map statuses to aggFilters format
            status_map = {
                "COMPLETED": "com",
                "RECRUITING": "rec",
                "NOT_YET_RECRUITING": "not",
                "ACTIVE_NOT_RECRUITING": "act",
            }
            for status in self.study_statuses:
                if status in status_map:
                    agg_filters.append(f"status:{status_map[status]}")

        if agg_filters:
            params["aggFilters"] = ",".join(agg_filters)

        return params


@dataclass
class CTSearchResult:
    """ClinicalTrials.gov search result."""

    nct_ids: list[str]
    total_count: int
    next_page_token: str | None = None


class ClinicalTrialsClient:
    """
    Client for ClinicalTrials.gov API v2.

    Implements:
        - Study search with filtering
        - Study metadata retrieval

    Usage:
        client = ClinicalTrialsClient()
        records = client.search_studies("diabetes GLP-1")
    """

    def __init__(
        self,
        max_results: int | None = None,
    ) -> None:
        """
        Initialize ClinicalTrials.gov client.

        Args:
            max_results: Maximum results per search.
        """
        settings = get_settings()
        self._max_results = max_results or settings.retrieval.clinical_trials_max_results
        self._client = httpx.Client(timeout=30.0)
        self._tracer = get_tracer("cdr.retrieval.clinicaltrials")

        # Rate limiting (be conservative)
        self._request_interval = 0.5
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request = time.time()

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize and shorten query for CT.gov API v2.

        CT.gov API v2 has strict limits on query.term length.
        This function:
        1. Removes excessive Boolean operators
        2. Extracts key terms
        3. Limits total length to 500 characters

        Args:
            query: Original query string.

        Returns:
            Sanitized query suitable for CT.gov API.
        """
        # Remove complex operators that CT.gov doesn't handle well
        sanitized = query.replace("[tiab]", "").replace("[mh]", "")
        sanitized = sanitized.replace("AND", " ").replace("OR", " ")
        sanitized = sanitized.replace("(", " ").replace(")", " ")
        sanitized = sanitized.replace('"', " ").replace("'", " ")

        # Split into words and take unique, meaningful terms
        words = sanitized.split()
        # Remove very short words and duplicates
        unique_words = []
        seen = set()
        for word in words:
            word_lower = word.lower().strip()
            if len(word_lower) >= 3 and word_lower not in seen:
                unique_words.append(word.strip())
                seen.add(word_lower)

        # Join back, limiting to 500 chars total
        result = " ".join(unique_words)
        if len(result) > 500:
            # Truncate at word boundary
            result = result[:500].rsplit(" ", 1)[0]

        return result

    def search(
        self,
        query: str,
        max_results: int | None = None,
        status: list[str] | None = None,
        study_type: str | None = None,
        filters: CTSearchFilters | None = None,
    ) -> CTSearchResult:
        """
        Search ClinicalTrials.gov.

        Args:
            query: Search query (will be sanitized).
            max_results: Maximum results.
            status: Filter by status (e.g., ["COMPLETED", "RECRUITING"]).
            study_type: Filter by study type (e.g., "INTERVENTIONAL").
            filters: Structured filters using CTSearchFilters object.

        Returns:
            Search result with NCT IDs.

        Documentation: https://clinicaltrials.gov/data-api/api
        """
        with self._tracer.span("search", attributes={"query": query}) as span:
            self._rate_limit()

            # Sanitize query to avoid 400 errors from CT.gov API
            sanitized_query = self._sanitize_query(query)
            span.set_attribute("original_query_length", len(query))
            span.set_attribute("sanitized_query_length", len(sanitized_query))

            if sanitized_query != query:
                print(f"[CT.gov] Query sanitized: {len(query)} -> {len(sanitized_query)} chars")

            params: dict[str, Any] = {
                "query.term": sanitized_query,
                "pageSize": max_results or self._max_results,
                "format": "json",
            }

            # Apply structured filters if provided
            if filters:
                params.update(filters.to_params())
            else:
                # Legacy filter parameters
                if status:
                    params["filter.overallStatus"] = ",".join(status)
                if study_type:
                    params["filter.studyType"] = study_type

            # Log applied filters
            filter_keys = [k for k in params.keys() if k.startswith("filter.")]
            if filter_keys:
                print(f"[CT.gov] Filters applied: {filter_keys}")

            try:
                response = self._client.get(f"{CT_API_BASE}/studies", params=params)
                response.raise_for_status()
                data = response.json()

                studies = data.get("studies", [])
                nct_ids = [
                    s["protocolSection"]["identificationModule"]["nctId"]
                    for s in studies
                    if "protocolSection" in s
                ]

                total_count = data.get("totalCount", len(nct_ids))
                next_page = data.get("nextPageToken")

                span.set_attribute("result_count", len(nct_ids))
                span.set_attribute("total_count", total_count)

                return CTSearchResult(
                    nct_ids=nct_ids,
                    total_count=total_count,
                    next_page_token=next_page,
                )

            except httpx.HTTPError as e:
                raise ClinicalTrialsError(f"Search failed: {e}") from e

    def fetch_study(self, nct_id: str) -> Record | None:
        """
        Fetch single study by NCT ID.

        Args:
            nct_id: NCT identifier (e.g., NCT01234567).

        Returns:
            Record object or None if not found.
        """
        with self._tracer.span("fetch_study", attributes={"nct_id": nct_id}) as span:
            self._rate_limit()

            try:
                response = self._client.get(
                    f"{CT_API_BASE}/studies/{nct_id}", params={"format": "json"}
                )

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                data = response.json()

                return self._parse_study(data)

            except httpx.HTTPError as e:
                span.add_event("fetch_error", {"error": str(e)})
                raise ClinicalTrialsError(f"Fetch failed for {nct_id}: {e}") from e

    def fetch_studies(self, nct_ids: list[str]) -> list[Record]:
        """
        Fetch multiple studies.

        Args:
            nct_ids: List of NCT IDs.

        Returns:
            List of Record objects.
        """
        records: list[Record] = []

        with self._tracer.span("fetch_studies", attributes={"count": len(nct_ids)}) as span:
            for nct_id in nct_ids:
                record = self.fetch_study(nct_id)
                if record:
                    records.append(record)

            span.set_attribute("fetched_count", len(records))

        return records

    def _parse_study(self, study: dict[str, Any]) -> Record | None:
        """Parse study JSON into Record."""
        try:
            protocol = study.get("protocolSection", {})
            if not protocol:
                return None

            # Identification
            id_module = protocol.get("identificationModule", {})
            nct_id = id_module.get("nctId")
            if not nct_id:
                return None

            title = id_module.get("officialTitle") or id_module.get("briefTitle", "No title")

            # Description
            desc_module = protocol.get("descriptionModule", {})
            abstract = desc_module.get("briefSummary") or desc_module.get("detailedDescription")

            # Status
            status_module = protocol.get("statusModule", {})
            overall_status = status_module.get("overallStatus")

            # Study design
            design_module = protocol.get("designModule", {})
            study_type_raw = design_module.get("studyType", "")
            phases = design_module.get("phases", [])

            # Conditions and interventions
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])

            interventions = []
            for intervention in protocol.get("armsInterventionsModule", {}).get(
                "interventions", []
            ):
                name = intervention.get("name", "")
                type_ = intervention.get("type", "")
                if name:
                    interventions.append(f"{type_}: {name}" if type_ else name)

            # Sponsors
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            sponsors = []
            lead = sponsor_module.get("leadSponsor", {})
            if lead.get("name"):
                sponsors.append(lead["name"])

            # Enrollment
            enrollment = design_module.get("enrollmentInfo", {}).get("count")

            # Dates
            start_date = status_module.get("startDateStruct", {}).get("date")

            # Compute content hash for deduplication
            content_hash = Record.compute_hash(title, abstract, None)

            return Record(
                record_id=f"ct_{nct_id}",
                source=RecordSource.CLINICAL_TRIALS,
                content_hash=content_hash,
                nct_id=nct_id,
                title=title,
                abstract=abstract,
            )

        except Exception as e:
            # Log error for debugging instead of silently returning None
            print(f"[CT.gov] Error parsing study: {e}")
            return None

    def _infer_study_type(
        self, study_type_raw: str, design_module: dict[str, Any]
    ) -> StudyType | None:
        """Infer StudyType from CT.gov data."""
        study_type_lower = study_type_raw.lower()

        if "interventional" in study_type_lower:
            # Check if randomized
            design_info = design_module.get("designInfo", {})
            allocation = design_info.get("allocation", "").lower()

            if "randomized" in allocation:
                return StudyType.RCT
            return StudyType.RCT  # Most interventional are RCTs

        if "observational" in study_type_lower:
            return StudyType.COHORT

        return None

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def search_rcts(
        self,
        query: str,
        max_results: int | None = None,
        include_completed: bool = True,
        include_recruiting: bool = True,
    ) -> list[Record]:
        """
        Search ClinicalTrials.gov specifically for Randomized Controlled Trials.

        This is a convenience method that applies appropriate filters for
        systematic reviews focused on RCTs.

        Args:
            query: Search query (will be sanitized).
            max_results: Maximum results.
            include_completed: Include completed trials (default True).
            include_recruiting: Include recruiting trials (default True).

        Returns:
            List of Record objects for interventional studies.

        Documentation: https://clinicaltrials.gov/data-api/api
        """
        # Build filter for RCTs
        statuses = []
        if include_completed:
            statuses.append("COMPLETED")
        if include_recruiting:
            statuses.extend(["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"])

        filters = CTSearchFilters(
            study_statuses=statuses,
            study_type="INTERVENTIONAL",  # RCTs are interventional
        )

        return self.search_studies(
            query=query,
            max_results=max_results,
            filters=filters,
        )

    def search_studies(
        self,
        query: str,
        max_results: int | None = None,
        filters: CTSearchFilters | None = None,
        **kwargs,
    ) -> list[Record]:
        """
        Search and fetch studies in one call.

        Args:
            query: Search query.
            max_results: Maximum results.
            filters: Structured filters using CTSearchFilters.
            **kwargs: Additional search parameters (legacy).

        Returns:
            List of Record objects.
        """
        with self._tracer.span("search_studies", attributes={"query": query}) as span:
            # Search with JSON response that includes study data
            self._rate_limit()

            # Sanitize query to avoid 400 errors from CT.gov API
            sanitized_query = self._sanitize_query(query)

            params: dict[str, Any] = {
                "query.term": sanitized_query,
                "pageSize": max_results or self._max_results,
                "format": "json",
            }

            # Apply structured filters
            if filters:
                params.update(filters.to_params())

            # Add legacy kwargs
            params.update(kwargs)

            try:
                response = self._client.get(f"{CT_API_BASE}/studies", params=params)
                response.raise_for_status()
                data = response.json()

                records: list[Record] = []
                for study in data.get("studies", []):
                    record = self._parse_study(study)
                    if record:
                        records.append(record)

                span.set_attribute("result_count", len(records))
                print(f"[CT.gov] Retrieved {len(records)} studies")
                return records

            except httpx.HTTPError as e:
                raise ClinicalTrialsError(f"Search failed: {e}") from e

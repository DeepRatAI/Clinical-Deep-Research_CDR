"""
PubMed Client

Client for NCBI E-utilities API to search PubMed/MEDLINE.

Implementation follows NCBI E-utilities best practices:
- ESearch: Search and retrieve PMIDs with history
- EFetch: Fetch article metadata in batches
- Rate limiting per NCBI guidelines (3/sec without key, 10/sec with key)

Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25501/
E-utilities reference: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import httpx

from cdr.config import get_settings
from cdr.core.enums import RecordSource, StudyType
from cdr.core.exceptions import PubMedError
from cdr.core.schemas import Record
from cdr.observability import get_tracer


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


# Common publication types for filtering
# See: https://www.ncbi.nlm.nih.gov/mesh/68016454
PUBLICATION_TYPES = {
    "RCT": "Randomized Controlled Trial",
    "META_ANALYSIS": "Meta-Analysis",
    "SYSTEMATIC_REVIEW": "Systematic Review",
    "CLINICAL_TRIAL": "Clinical Trial",
    "REVIEW": "Review",
    "CASE_REPORTS": "Case Reports",
    "GUIDELINE": "Practice Guideline",
}


@dataclass
class PubMedSearchFilters:
    """Structured filters for PubMed search.

    Documentation: https://www.ncbi.nlm.nih.gov/books/NBK3827/#pubmedhelp.Publication_Types_Scope_Not
    """

    publication_types: list[str] = field(default_factory=list)
    date_from: str | None = None  # YYYY/MM/DD
    date_to: str | None = None  # YYYY/MM/DD
    humans_only: bool = False
    english_only: bool = False

    def to_filter_string(self) -> str:
        """Convert filters to PubMed filter string."""
        filters = []
        if self.publication_types:
            pt_filter = " OR ".join(f'"{pt}"[pt]' for pt in self.publication_types)
            filters.append(f"({pt_filter})")
        if self.humans_only:
            filters.append('"Humans"[mh]')
        if self.english_only:
            filters.append('"English"[la]')
        return " AND ".join(filters) if filters else ""


@dataclass
class PubMedSearchResult:
    """PubMed search result."""

    pmids: list[str]
    total_count: int
    query_translation: str | None = None
    web_env: str | None = None
    query_key: str | None = None


class PubMedClient:
    """
    Client for PubMed E-utilities.

    Implements:
        - ESearch: Search and retrieve PMIDs
        - EFetch: Fetch article metadata
        - Rate limiting per NCBI guidelines

    Usage:
        client = PubMedClient()
        pmids = client.search("diabetes AND GLP-1 agonists")
        records = client.fetch_records(pmids)
    """

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        max_results: int | None = None,
    ) -> None:
        """
        Initialize PubMed client.

        Args:
            api_key: NCBI API key (increases rate limit).
            email: Contact email (required by NCBI).
            max_results: Maximum results per search.
        """
        settings = get_settings()
        self._api_key = api_key or settings.retrieval.ncbi_api_key
        self._email = email or settings.retrieval.ncbi_email
        self._max_results = max_results or settings.retrieval.pubmed_max_results

        # Rate limiting: 3/sec without key, 10/sec with key
        self._request_interval = 0.1 if self._api_key else 0.34
        self._last_request = 0.0

        self._client = httpx.Client(timeout=30.0)
        self._tracer = get_tracer("cdr.retrieval.pubmed")

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request = time.time()

    def _build_params(self, **kwargs) -> dict[str, str]:
        """Build request parameters."""
        params = {"db": "pubmed", "retmode": "xml"}
        if self._api_key:
            params["api_key"] = self._api_key
        if self._email:
            params["email"] = self._email
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return params

    def search(
        self,
        query: str,
        max_results: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        publication_types: list[str] | None = None,
        filters: PubMedSearchFilters | None = None,
    ) -> PubMedSearchResult:
        """
        Search PubMed and return PMIDs.

        Uses ESearch E-utility with history for reproducible results.

        Args:
            query: PubMed query string.
            max_results: Override default max results.
            date_from: Start date (YYYY/MM/DD).
            date_to: End date (YYYY/MM/DD).
            publication_types: Filter by publication type (legacy).
            filters: Structured filters using PubMedSearchFilters.

        Returns:
            Search result with PMIDs and metadata.

        Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25499/
        """
        with self._tracer.span("search", attributes={"query": query}) as span:
            self._rate_limit()

            # Build query with filters
            full_query = query

            # Apply structured filters if provided
            if filters:
                filter_str = filters.to_filter_string()
                if filter_str:
                    full_query = f"({query}) AND {filter_str}"
                # Use filter dates if not explicitly provided
                if filters.date_from and not date_from:
                    date_from = filters.date_from
                if filters.date_to and not date_to:
                    date_to = filters.date_to
            elif publication_types:
                # Legacy filter parameters
                pt_filter = " OR ".join(f'"{pt}"[pt]' for pt in publication_types)
                full_query = f"({query}) AND ({pt_filter})"

            # Log the full query for reproducibility
            print(f"[PubMed] Full query: {full_query}")
            span.set_attribute("full_query", full_query)

            params = self._build_params(
                term=full_query,
                retmax=str(max_results or self._max_results),
                usehistory="y",
                mindate=date_from,
                maxdate=date_to,
                datetype="pdat" if (date_from or date_to) else None,
            )

            try:
                response = self._client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
                response.raise_for_status()

                root = ET.fromstring(response.text)

                # Check for errors
                error = root.find(".//ErrorList/PhraseNotFound")
                if error is not None:
                    span.add_event("search_warning", {"phrase_not_found": error.text})

                # Extract results
                count_elem = root.find("Count")
                total_count = int(count_elem.text) if count_elem is not None else 0

                pmids = [id_elem.text for id_elem in root.findall(".//IdList/Id") if id_elem.text]

                query_translation = root.findtext("QueryTranslation")
                web_env = root.findtext("WebEnv")
                query_key = root.findtext("QueryKey")

                span.set_attribute("result_count", len(pmids))
                span.set_attribute("total_count", total_count)

                return PubMedSearchResult(
                    pmids=pmids,
                    total_count=total_count,
                    query_translation=query_translation,
                    web_env=web_env,
                    query_key=query_key,
                )

            except httpx.HTTPError as e:
                raise PubMedError(f"Search failed: {e}") from e
            except ET.ParseError as e:
                raise PubMedError(f"Failed to parse response: {e}") from e

    def fetch_records(self, pmids: list[str], batch_size: int = 100) -> list[Record]:
        """
        Fetch article metadata for PMIDs.

        Args:
            pmids: List of PubMed IDs.
            batch_size: Batch size for requests.

        Returns:
            List of Record objects.
        """
        if not pmids:
            return []

        with self._tracer.span("fetch_records", attributes={"count": len(pmids)}) as span:
            records: list[Record] = []

            for i in range(0, len(pmids), batch_size):
                batch = pmids[i : i + batch_size]
                self._rate_limit()

                params = self._build_params(
                    id=",".join(batch),
                    rettype="xml",
                    retmode="xml",
                )

                try:
                    response = self._client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
                    response.raise_for_status()

                    batch_records = self._parse_efetch_response(response.text)
                    records.extend(batch_records)

                except httpx.HTTPError as e:
                    span.add_event("batch_error", {"batch_start": i, "error": str(e)})
                    raise PubMedError(f"Fetch failed for batch {i}: {e}") from e

            span.set_attribute("fetched_count", len(records))
            return records

    def _parse_efetch_response(self, xml_text: str) -> list[Record]:
        """Parse EFetch XML response into Record objects."""
        records: list[Record] = []

        try:
            root = ET.fromstring(xml_text)
            articles = root.findall(".//PubmedArticle")
            print(f"[PubMed] Found {len(articles)} articles in XML response")

            for article in articles:
                try:
                    record = self._parse_article(article)
                    if record:
                        records.append(record)
                except Exception as e:
                    pmid = article.find(".//PMID")
                    pmid_text = pmid.text if pmid is not None else "unknown"
                    print(f"[PubMed] Failed to parse article {pmid_text}: {e}")

        except ET.ParseError as e:
            raise PubMedError(f"Failed to parse XML: {e}") from e

        print(f"[PubMed] Successfully parsed {len(records)} records")
        return records

    def _parse_article(self, article: ET.Element) -> Record | None:
        """Parse single article element."""
        try:
            # Get PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text

            # Get citation
            citation = article.find(".//MedlineCitation")
            if citation is None:
                return None

            article_elem = citation.find("Article")
            if article_elem is None:
                return None

            # Title
            title_elem = article_elem.find("ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"

            # Abstract
            abstract_parts = []
            for abstract in article_elem.findall(".//AbstractText"):
                label = abstract.get("Label", "")
                text = abstract.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                if last:
                    authors.append(f"{last} {initials or fore}".strip())

            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else None

            # Year
            year = None
            pub_date = article_elem.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    year = int(year_elem.text)

            # DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break

            # Study type inference from publication types
            study_type = self._infer_study_type(article)

            # Compute content hash for deduplication
            content_hash = Record.compute_hash(title, abstract, doi)

            return Record(
                record_id=f"pubmed_{pmid}",
                source=RecordSource.PUBMED,
                content_hash=content_hash,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                pmid=pmid,
                mesh_terms=self._extract_mesh_terms(article),
                keywords=self._extract_keywords(article),
            )

        except Exception as e:
            print(f"[PubMed] _parse_article exception: {e}")
            return None

    def _infer_study_type(self, article: ET.Element) -> StudyType | None:
        """Infer study type from publication types."""
        pub_types = [pt.text.lower() for pt in article.findall(".//PublicationType") if pt.text]

        if "meta-analysis" in pub_types:
            return StudyType.META_ANALYSIS
        if "systematic review" in pub_types:
            return StudyType.SYSTEMATIC_REVIEW
        if "randomized controlled trial" in pub_types:
            return StudyType.RCT
        if "clinical trial" in pub_types:
            return StudyType.RCT
        if "observational study" in pub_types:
            return StudyType.COHORT
        if "case reports" in pub_types:
            return StudyType.CASE_REPORT

        return None

    def _extract_mesh_terms(self, article: ET.Element) -> list[str]:
        """Extract MeSH terms."""
        terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                terms.append(mesh.text)
        return terms

    def _extract_keywords(self, article: ET.Element) -> list[str]:
        """Extract keywords."""
        keywords = []
        for kw in article.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)
        return keywords

    def search_and_fetch(
        self, query: str, max_results: int | None = None, **kwargs
    ) -> list[Record]:
        """
        Convenience method: search and fetch in one call.

        Args:
            query: PubMed query.
            max_results: Maximum results.
            **kwargs: Additional search parameters.

        Returns:
            List of Record objects.
        """
        result = self.search(query, max_results=max_results, **kwargs)
        return self.fetch_records(result.pmids)

    def search_rcts(
        self,
        query: str,
        max_results: int | None = None,
        humans_only: bool = True,
    ) -> list[Record]:
        """
        Search PubMed specifically for Randomized Controlled Trials.

        This is a convenience method that applies appropriate filters for
        systematic reviews focused on RCTs.

        Args:
            query: PubMed query string.
            max_results: Maximum results.
            humans_only: Filter to human studies only (default True).

        Returns:
            List of Record objects for RCTs.

        Documentation: https://www.ncbi.nlm.nih.gov/books/NBK3827/#pubmedhelp.Publication_Types_Scope_Not
        """
        filters = PubMedSearchFilters(
            publication_types=["Randomized Controlled Trial"],
            humans_only=humans_only,
        )

        result = self.search(
            query=query,
            max_results=max_results,
            filters=filters,
        )

        print(f"[PubMed] RCT search found {result.total_count} total, fetching {len(result.pmids)}")
        return self.fetch_records(result.pmids)

    def search_systematic_reviews(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[Record]:
        """
        Search PubMed for Systematic Reviews and Meta-Analyses.

        Useful for finding existing reviews on a topic.

        Args:
            query: PubMed query string.
            max_results: Maximum results.

        Returns:
            List of Record objects for systematic reviews.
        """
        filters = PubMedSearchFilters(
            publication_types=["Systematic Review", "Meta-Analysis"],
        )

        result = self.search(
            query=query,
            max_results=max_results,
            filters=filters,
        )

        print(
            f"[PubMed] SR/MA search found {result.total_count} total, fetching {len(result.pmids)}"
        )
        return self.fetch_records(result.pmids)

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()

    def __enter__(self) -> "PubMedClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

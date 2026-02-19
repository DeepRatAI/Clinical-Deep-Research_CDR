"""
PMC Full-Text Retrieval Client

Client for retrieving full-text articles from PubMed Central (PMC).

Implementation follows NCBI E-utilities and PMC OA Service guidelines:
- ID Converter: Convert PMID to PMCID for PMC lookup
- PMC OAI-PMH: Retrieve full-text XML for open access articles
- Fallback: Abstract-only if full-text unavailable

Documentation:
- PMC OAI-PMH API: https://pmc.ncbi.nlm.nih.gov/tools/oai/
- ID Converter: https://www.ncbi.nlm.nih.gov/pmc/tools/idconv/
- PMC OA Service: https://www.ncbi.nlm.nih.gov/pmc/tools/oa-service/

HIGH-2 fix: Full-text fallback before reports_not_retrieved
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-2, PRISMA 2020
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import httpx

from cdr.config import get_settings
from cdr.observability import get_tracer


tracer = get_tracer(__name__)

# NCBI PMC API endpoints
PMC_OAI_BASE = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
ID_CONVERTER_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Rate limiting: 3 requests/sec without API key, 10/sec with key
# Per NCBI guidelines: https://www.ncbi.nlm.nih.gov/books/NBK25497/
MIN_REQUEST_INTERVAL = 0.35  # ~3/sec baseline


@dataclass
class FullTextResult:
    """Result of full-text retrieval attempt.

    Attributes:
        record_id: Original CDR record ID
        pmid: PubMed ID if available
        pmcid: PMC ID if found
        full_text: Full-text content if retrieved
        sections: Extracted sections (abstract, methods, results, etc.)
        source: Where the text came from ('pmc_fulltext', 'abstract_fallback')
        retrieval_reason: Why this source was used
        is_open_access: Whether article is in PMC OA subset
    """

    record_id: str
    pmid: str | None = None
    pmcid: str | None = None
    full_text: str | None = None
    sections: dict[str, str] | None = None
    source: Literal["pmc_fulltext", "abstract_fallback", "not_retrieved"] = "not_retrieved"
    retrieval_reason: str = ""
    is_open_access: bool = False


class FullTextClient:
    """
    Client for retrieving full-text articles from PMC.

    Implements fallback chain:
    1. Check if PMID has PMCID (via ID converter)
    2. If PMCID exists, check if article is in OA subset
    3. If OA, retrieve full-text XML
    4. Parse XML to extract sections
    5. Fallback to abstract if any step fails

    Usage:
        client = FullTextClient()
        result = await client.get_full_text(record)
    """

    def __init__(
        self,
        api_key: str | None = None,
        tool_name: str = "CDR-ClinicalDeepResearch",
        tool_email: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize PMC client.

        Args:
            api_key: NCBI API key for higher rate limits
            tool_name: Tool name for NCBI tracking (required)
            tool_email: Email for NCBI tracking (recommended)
            timeout: HTTP request timeout in seconds
        """
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "ncbi_api_key", None)
        self.tool_name = tool_name
        self.tool_email = tool_email or getattr(settings, "ncbi_email", None)
        self.timeout = timeout
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting per NCBI guidelines."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    async def _get_pmcid_from_pmid(self, pmid: str) -> str | None:
        """Convert PMID to PMCID using ID Converter API.

        Docs: https://www.ncbi.nlm.nih.gov/pmc/tools/idconv/

        Returns:
            PMCID string (e.g., "PMC1234567") or None if not in PMC
        """
        with tracer.start_span("fulltext.id_convert") as span:
            span.set_attribute("pmid", pmid)

            self._rate_limit()

            params = {
                "ids": pmid,
                "format": "json",
                "tool": self.tool_name,
            }
            if self.tool_email:
                params["email"] = self.tool_email
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(ID_CONVERTER_BASE, params=params)
                    response.raise_for_status()

                    data = response.json()
                    records = data.get("records", [])
                    if records and "pmcid" in records[0]:
                        pmcid = records[0]["pmcid"]
                        span.set_attribute("pmcid", pmcid)
                        return pmcid
                    return None

            except Exception as e:
                span.set_attribute("error", str(e))
                return None

    async def _check_oa_availability(self, pmcid: str) -> dict | None:
        """Check if article is in PMC Open Access subset.

        Docs: https://www.ncbi.nlm.nih.gov/pmc/tools/oa-service/

        Returns:
            Dict with OA info (tgz/pdf URLs) or None if not available
        """
        with tracer.start_span("fulltext.oa_check") as span:
            span.set_attribute("pmcid", pmcid)

            self._rate_limit()

            # OA service uses numeric PMCID without prefix
            pmcid_num = pmcid.replace("PMC", "")
            params = {"id": f"PMC{pmcid_num}"}

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(PMC_OA_SERVICE, params=params)
                    response.raise_for_status()

                    # Parse XML response
                    root = ET.fromstring(response.text)

                    # Check for error
                    error = root.find(".//error")
                    if error is not None:
                        span.set_attribute("oa_error", error.text)
                        return None

                    # Find record
                    record = root.find(".//record")
                    if record is None:
                        return None

                    # Extract links
                    links = {}
                    for link in record.findall(".//link"):
                        format_type = link.get("format")
                        href = link.get("href")
                        if format_type and href:
                            links[format_type] = href

                    span.set_attribute("oa_formats", list(links.keys()))
                    return links if links else None

            except Exception as e:
                span.set_attribute("error", str(e))
                return None

    async def _fetch_fulltext_xml(self, pmcid: str) -> str | None:
        """Fetch full-text XML from PMC OAI-PMH API.

        Docs: https://pmc.ncbi.nlm.nih.gov/tools/oai/

        Uses GetRecord verb with metadataPrefix=pmc for full text.

        Returns:
            Full-text XML string or None if unavailable
        """
        with tracer.start_span("fulltext.fetch_xml") as span:
            span.set_attribute("pmcid", pmcid)

            self._rate_limit()

            # OAI identifier format: oai:pubmedcentral.nih.gov:NNNNNNN
            pmcid_num = pmcid.replace("PMC", "")
            oai_id = f"oai:pubmedcentral.nih.gov:{pmcid_num}"

            params = {
                "verb": "GetRecord",
                "identifier": oai_id,
                "metadataPrefix": "pmc",  # Full text format
            }

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        PMC_OAI_BASE,
                        params=params,
                        headers={"Accept-Encoding": "gzip, deflate"},
                    )
                    response.raise_for_status()

                    xml_content = response.text
                    span.set_attribute("xml_length", len(xml_content))
                    return xml_content

            except Exception as e:
                span.set_attribute("error", str(e))
                return None

    def _parse_jats_xml(self, xml_content: str) -> dict[str, str]:
        """Parse JATS XML to extract article sections.

        JATS (Journal Article Tag Suite) is the standard for PMC articles.
        Docs: https://jats.nlm.nih.gov/archiving/

        Returns:
            Dict mapping section name to text content
        """
        sections = {}

        try:
            root = ET.fromstring(xml_content)

            # Namespace handling for OAI-PMH wrapper
            # The actual article is inside <metadata><article>
            ns = {
                "oai": "http://www.openarchives.org/OAI/2.0/",
            }

            # Find article element (may be wrapped in OAI envelope)
            article = root.find(".//article")
            if article is None:
                # Try with namespace
                article = root.find(".//oai:metadata//article", ns)
            if article is None:
                return sections

            # Extract abstract
            abstract_elem = article.find(".//abstract")
            if abstract_elem is not None:
                abstract_text = "".join(abstract_elem.itertext()).strip()
                if abstract_text:
                    sections["abstract"] = abstract_text

            # Extract body sections
            body = article.find(".//body")
            if body is not None:
                for sec in body.findall(".//sec"):
                    title_elem = sec.find("title")
                    title = (
                        title_elem.text.strip().lower()
                        if title_elem is not None and title_elem.text
                        else "body"
                    )

                    # Extract paragraph text
                    paragraphs = []
                    for p in sec.findall(".//p"):
                        p_text = "".join(p.itertext()).strip()
                        if p_text:
                            paragraphs.append(p_text)

                    if paragraphs:
                        # Map common section titles to standard names
                        section_map = {
                            "introduction": "introduction",
                            "background": "introduction",
                            "methods": "methods",
                            "materials and methods": "methods",
                            "methodology": "methods",
                            "results": "results",
                            "findings": "results",
                            "discussion": "discussion",
                            "conclusions": "conclusions",
                            "conclusion": "conclusions",
                        }
                        section_name = section_map.get(title, title)

                        if section_name in sections:
                            sections[section_name] += "\n\n" + "\n\n".join(paragraphs)
                        else:
                            sections[section_name] = "\n\n".join(paragraphs)

        except ET.ParseError:
            pass

        return sections

    async def get_full_text(
        self,
        record_id: str,
        pmid: str | None = None,
        abstract: str | None = None,
    ) -> FullTextResult:
        """Attempt to retrieve full-text for a record.

        Implements fallback chain:
        1. If PMID, try to get PMCID
        2. If PMCID, check OA availability
        3. If OA, fetch full-text XML
        4. Parse sections from XML
        5. Fallback to abstract if any step fails

        Args:
            record_id: CDR record ID
            pmid: PubMed ID if available
            abstract: Fallback abstract text

        Returns:
            FullTextResult with full_text, sections, and source info
        """
        with tracer.start_span("fulltext.get_full_text") as span:
            span.set_attribute("record_id", record_id)
            span.set_attribute("has_pmid", pmid is not None)

            result = FullTextResult(record_id=record_id, pmid=pmid)

            # Step 1: Try to get PMCID from PMID
            if pmid:
                pmcid = await self._get_pmcid_from_pmid(pmid)
                result.pmcid = pmcid

                if pmcid:
                    span.set_attribute("pmcid", pmcid)

                    # Step 2: Check OA availability
                    oa_info = await self._check_oa_availability(pmcid)

                    if oa_info:
                        result.is_open_access = True
                        span.set_attribute("is_open_access", True)

                        # Step 3: Fetch full-text XML
                        xml_content = await self._fetch_fulltext_xml(pmcid)

                        if xml_content and len(xml_content) > 100:
                            # Step 4: Parse sections
                            sections = self._parse_jats_xml(xml_content)

                            if sections:
                                # Combine sections into full text
                                full_text_parts = []
                                for sec_name in [
                                    "abstract",
                                    "introduction",
                                    "methods",
                                    "results",
                                    "discussion",
                                    "conclusions",
                                ]:
                                    if sec_name in sections:
                                        full_text_parts.append(
                                            f"## {sec_name.title()}\n{sections[sec_name]}"
                                        )

                                # Add any remaining sections
                                for sec_name, sec_text in sections.items():
                                    if sec_name not in [
                                        "abstract",
                                        "introduction",
                                        "methods",
                                        "results",
                                        "discussion",
                                        "conclusions",
                                    ]:
                                        full_text_parts.append(f"## {sec_name.title()}\n{sec_text}")

                                if full_text_parts:
                                    result.full_text = "\n\n".join(full_text_parts)
                                    result.sections = sections
                                    result.source = "pmc_fulltext"
                                    result.retrieval_reason = (
                                        f"Full text retrieved from PMC ({pmcid})"
                                    )
                                    span.set_attribute("source", "pmc_fulltext")
                                    span.set_attribute("sections_count", len(sections))
                                    return result

                        result.retrieval_reason = (
                            f"PMCID {pmcid} exists but full-text XML parsing failed"
                        )
                    else:
                        result.retrieval_reason = f"PMCID {pmcid} exists but not in OA subset"
                else:
                    result.retrieval_reason = "PMID not in PMC"
            else:
                result.retrieval_reason = "No PMID available for PMC lookup"

            # Fallback to abstract
            if abstract and len(abstract) >= 10:
                result.full_text = abstract
                result.sections = {"abstract": abstract}
                result.source = "abstract_fallback"
                span.set_attribute("source", "abstract_fallback")
                return result

            # No content available
            result.source = "not_retrieved"
            if not result.retrieval_reason:
                result.retrieval_reason = "No abstract and no PMC full-text available"
            span.set_attribute("source", "not_retrieved")
            return result

"""
Document Parser

Parsing PDF and other documents using Unstructured and PyMuPDF.
Preserves document structure for accurate citation extraction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cdr.core.enums import Section
from cdr.core.exceptions import ParsingError
from cdr.observability import get_tracer


@dataclass
class ParsedElement:
    """A parsed document element."""

    text: str
    element_type: str  # title, narrative_text, table, figure, list_item, etc.
    section: Section | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Parsed document with structure."""

    source_path: str
    elements: list[ParsedElement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_section_text(self, section: Section) -> str:
        """Get all text from a specific section."""
        texts = [e.text for e in self.elements if e.section == section]
        return "\n\n".join(texts)

    def get_full_text(self) -> str:
        """Get complete document text."""
        return "\n\n".join(e.text for e in self.elements)

    def get_text_by_page(self, page: int) -> str:
        """Get text from a specific page."""
        texts = [e.text for e in self.elements if e.page == page]
        return "\n\n".join(texts)


# Section detection patterns
SECTION_PATTERNS = {
    Section.ABSTRACT: ["abstract", "summary", "synopsis"],
    Section.INTRODUCTION: ["introduction", "background", "rationale"],
    Section.METHODS: [
        "methods",
        "methodology",
        "materials and methods",
        "study design",
        "participants",
    ],
    Section.RESULTS: ["results", "findings", "outcomes"],
    Section.DISCUSSION: ["discussion", "interpretation"],
    Section.CONCLUSION: ["conclusion", "conclusions", "summary"],
    Section.REFERENCES: ["references", "bibliography", "literature cited"],
    Section.SUPPLEMENTARY: ["supplementary", "supplemental", "appendix", "appendices"],
}


def _detect_section(text: str) -> Section | None:
    """Detect section from header text."""
    text_lower = text.lower().strip()
    for section, patterns in SECTION_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            return section
    return None


class PDFParser:
    """
    PDF parser using PyMuPDF (fitz).

    Usage:
        parser = PDFParser()
        doc = parser.parse("/path/to/file.pdf")
    """

    def __init__(self) -> None:
        """Initialize PDF parser."""
        self._tracer = get_tracer("cdr.parsing.pdf")

        try:
            import fitz

            self._fitz = fitz
        except ImportError:
            raise ParsingError("PyMuPDF not installed. Run: pip install pymupdf")

    def parse(self, path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            path: Path to PDF file.

        Returns:
            Parsed document.
        """
        path = Path(path)

        if not path.exists():
            raise ParsingError(f"File not found: {path}")

        with self._tracer.span("parse_pdf", attributes={"path": str(path)}) as span:
            try:
                doc = self._fitz.open(path)
                elements: list[ParsedElement] = []
                current_section: Section | None = None

                for page_num in range(len(doc)):
                    page = doc[page_num]
                    blocks = page.get_text("dict")["blocks"]

                    for block in blocks:
                        if block.get("type") != 0:  # 0 = text block
                            continue

                        for line in block.get("lines", []):
                            text = "".join(span["text"] for span in line.get("spans", [])).strip()

                            if not text:
                                continue

                            # Detect if this is a section header
                            detected_section = _detect_section(text)
                            if detected_section:
                                current_section = detected_section
                                elements.append(
                                    ParsedElement(
                                        text=text,
                                        element_type="title",
                                        section=current_section,
                                        page=page_num + 1,
                                    )
                                )
                            else:
                                # Regular text
                                elements.append(
                                    ParsedElement(
                                        text=text,
                                        element_type="narrative_text",
                                        section=current_section,
                                        page=page_num + 1,
                                    )
                                )

                doc.close()

                span.set_attribute("page_count", len(doc))
                span.set_attribute("element_count", len(elements))

                return ParsedDocument(
                    source_path=str(path),
                    elements=elements,
                    metadata={"page_count": len(doc)},
                )

            except Exception as e:
                raise ParsingError(f"Failed to parse PDF: {e}") from e

    def extract_text(self, path: str | Path) -> str:
        """Extract plain text from PDF."""
        doc = self.parse(path)
        return doc.get_full_text()


class UnstructuredParser:
    """
    Document parser using Unstructured library.

    Supports multiple formats: PDF, DOCX, HTML, etc.
    """

    def __init__(self) -> None:
        """Initialize parser."""
        self._tracer = get_tracer("cdr.parsing.unstructured")

        try:
            from unstructured.partition.auto import partition

            self._partition = partition
        except ImportError:
            raise ParsingError("unstructured not installed. Run: pip install unstructured")

    def parse(self, path: str | Path) -> ParsedDocument:
        """
        Parse a document file.

        Args:
            path: Path to document.

        Returns:
            Parsed document.
        """
        path = Path(path)

        if not path.exists():
            raise ParsingError(f"File not found: {path}")

        with self._tracer.span("parse_unstructured", attributes={"path": str(path)}) as span:
            try:
                elements_raw = self._partition(filename=str(path))

                elements: list[ParsedElement] = []
                current_section: Section | None = None

                for elem in elements_raw:
                    text = str(elem)
                    if not text.strip():
                        continue

                    elem_type = type(elem).__name__.lower()

                    # Check for section headers
                    if elem_type == "title":
                        detected_section = _detect_section(text)
                        if detected_section:
                            current_section = detected_section

                    # Get page number if available
                    page = None
                    if hasattr(elem, "metadata") and hasattr(elem.metadata, "page_number"):
                        page = elem.metadata.page_number

                    elements.append(
                        ParsedElement(
                            text=text,
                            element_type=elem_type,
                            section=current_section,
                            page=page,
                        )
                    )

                span.set_attribute("element_count", len(elements))

                return ParsedDocument(
                    source_path=str(path),
                    elements=elements,
                )

            except Exception as e:
                raise ParsingError(f"Failed to parse document: {e}") from e


class Parser:
    """
    High-level document parser that selects appropriate backend.
    """

    def __init__(self, prefer_unstructured: bool = False) -> None:
        """
        Initialize parser.

        Args:
            prefer_unstructured: Use Unstructured for all files.
        """
        self._prefer_unstructured = prefer_unstructured
        self._pdf_parser: PDFParser | None = None
        self._unstructured_parser: UnstructuredParser | None = None

    def _get_pdf_parser(self) -> PDFParser:
        """Lazily initialize PDF parser."""
        if self._pdf_parser is None:
            self._pdf_parser = PDFParser()
        return self._pdf_parser

    def _get_unstructured_parser(self) -> UnstructuredParser:
        """Lazily initialize Unstructured parser."""
        if self._unstructured_parser is None:
            self._unstructured_parser = UnstructuredParser()
        return self._unstructured_parser

    def parse(self, path: str | Path) -> ParsedDocument:
        """
        Parse a document.

        Args:
            path: Path to document.

        Returns:
            Parsed document.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if self._prefer_unstructured:
            return self._get_unstructured_parser().parse(path)

        if suffix == ".pdf":
            return self._get_pdf_parser().parse(path)
        else:
            return self._get_unstructured_parser().parse(path)

    def parse_bytes(
        self, content: bytes, filename: str, temp_dir: Path | None = None
    ) -> ParsedDocument:
        """
        Parse document from bytes.

        Args:
            content: Document bytes.
            filename: Original filename (for format detection).
            temp_dir: Temporary directory for file.

        Returns:
            Parsed document.
        """
        import tempfile

        temp_dir = temp_dir or Path(tempfile.gettempdir())
        temp_path = temp_dir / filename

        try:
            temp_path.write_bytes(content)
            return self.parse(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

"""
CDR Parsing Layer

Document parsing with structure preservation.
"""

from cdr.parsing.parser import (
    ParsedDocument,
    ParsedElement,
    Parser,
    PDFParser,
    UnstructuredParser,
)

__all__ = [
    "Parser",
    "PDFParser",
    "UnstructuredParser",
    "ParsedDocument",
    "ParsedElement",
]

import io
import re

from pypdf import PdfReader

_WHITESPACE = re.compile(r"\s+")


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return (page_number, normalized_text) for every non-empty page."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = _WHITESPACE.sub(" ", raw).strip()
        if text:
            pages.append((i, text))
    return pages
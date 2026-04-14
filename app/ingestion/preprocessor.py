"""Text preprocessing / cleaning utilities."""

from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document


def preprocess(docs: list[Document]) -> list[Document]:
    """Clean and normalise a batch of documents in-place and return them."""
    cleaned: list[Document] = []
    for doc in docs:
        text = doc.page_content
        text = _normalize_unicode(text)
        text = _strip_boilerplate(text)
        text = _collapse_whitespace(text)
        if text.strip():
            doc.page_content = text.strip()
            cleaned.append(doc)
    return cleaned


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _strip_boilerplate(text: str) -> str:
    """Remove common PDF header/footer noise."""
    # Page numbers like "Page 3 of 10", "- 3 -"
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"^-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
    # Repeated dashes / underscores (decorative lines)
    text = re.sub(r"[-_=]{10,}", "", text)
    return text


def _collapse_whitespace(text: str) -> str:
    """Normalise runs of whitespace while keeping paragraph breaks."""
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[^\S\n]+", " ", text)
    return text

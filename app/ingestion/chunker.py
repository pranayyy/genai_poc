"""Chunking strategies for documents."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks.

    Structured data rows (``source_type`` in ``{"csv", "json"}``) are kept
    atomic — one chunk per row — with schema context prepended.
    Everything else uses ``RecursiveCharacterTextSplitter``.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    unstructured: list[Document] = []
    structured: list[Document] = []

    for doc in docs:
        st = doc.metadata.get("source_type", "")
        if st in ("csv", "json"):
            structured.append(doc)
        else:
            unstructured.append(doc)

    chunks: list[Document] = []
    chunks.extend(_chunk_unstructured(unstructured, chunk_size, chunk_overlap))
    chunks.extend(_chunk_structured(structured))

    # Assign chunk_id and chunk_index
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = _make_chunk_id(chunk.page_content, idx)

    return chunks


# ── Private helpers ─────────────────────────────────────────


def _chunk_unstructured(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _chunk_structured(docs: list[Document]) -> list[Document]:
    """Each structured-data row is kept as a single chunk.

    We prepend a short schema line so the embedding captures what
    the fields represent.
    """
    chunks: list[Document] = []
    for doc in docs:
        try:
            data = json.loads(doc.page_content)
        except (json.JSONDecodeError, TypeError):
            data = None

        if isinstance(data, dict):
            schema_line = "Fields: " + ", ".join(data.keys())
            text = f"{schema_line}\n{doc.page_content}"
        else:
            text = doc.page_content

        chunks.append(Document(page_content=text, metadata={**doc.metadata}))
    return chunks


def _make_chunk_id(content: str, index: int) -> str:
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"chunk-{index}-{h}"

"""Dense retriever backed by ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.retrieval import vector_store


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


def retrieve(
    query_text: str,
    top_k: int | None = None,
    source_type: str | None = None,
    source: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from the vector store.

    Optionally filter by *source_type* or *source*.
    """
    top_k = top_k or settings.retrieval_top_k

    where_filter: dict[str, Any] | None = None
    if source_type or source:
        conditions = []
        if source_type:
            conditions.append({"source_type": {"$eq": source_type}})
        if source:
            conditions.append({"source": {"$eq": source}})
        where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    try:
        results: list[tuple[Document, float]] = vector_store.query_with_scores(
            query_text, top_k=top_k, where_filter=where_filter
        )
    except Exception as exc:
        # ChromaDB raises "Nothing found on disk" when the collection is empty
        # (HNSW index not yet built). Treat as zero results instead of crashing.
        _EMPTY_DB_PHRASES = ("nothing found on disk", "hnsw segment")
        msg = str(exc).lower()
        if any(p in msg for p in _EMPTY_DB_PHRASES):
            return []
        raise

    return [
        RetrievedChunk(
            content=doc.page_content,
            metadata=doc.metadata,
            score=score,
        )
        for doc, score in results
    ]

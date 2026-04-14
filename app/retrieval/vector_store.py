"""ChromaDB vector store wrapper."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.embedder import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return a persistent Chroma collection backed by the configured directory.

    Cached so the same ChromaDB client and httpx connection pool are reused
    across all calls within a process — avoids 'client has been closed' errors
    during multi-query evaluation loops.
    """
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(docs: list[Document], batch_size: int | None = None) -> None:
    """Embed and upsert *docs* into ChromaDB in batches."""
    batch_size = batch_size or settings.embedding_batch_size
    store = get_vector_store()
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch)


def query(
    query_text: str,
    top_k: int | None = None,
    where_filter: dict[str, Any] | None = None,
) -> list[Document]:
    """Retrieve the most similar documents for *query_text*."""
    top_k = top_k or settings.retrieval_top_k
    store = get_vector_store()
    kwargs: dict[str, Any] = {"k": top_k}
    if where_filter:
        kwargs["filter"] = where_filter
    return store.similarity_search(query_text, **kwargs)


def query_with_scores(
    query_text: str,
    top_k: int | None = None,
    where_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents together with their similarity scores."""
    top_k = top_k or settings.retrieval_top_k
    store = get_vector_store()
    kwargs: dict[str, Any] = {"k": top_k}
    if where_filter:
        kwargs["filter"] = where_filter
    return store.similarity_search_with_relevance_scores(query_text, **kwargs)


def delete_collection() -> None:
    """Drop the entire collection."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    client.delete_collection(settings.chroma_collection_name)


def get_stats() -> dict[str, Any]:
    """Return basic collection stats."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(settings.chroma_collection_name)
        return {"collection": settings.chroma_collection_name, "count": col.count()}
    except Exception:
        return {"collection": settings.chroma_collection_name, "count": 0}

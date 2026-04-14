"""Tests for retrieval (vector store operations)."""

from __future__ import annotations

import tempfile

import pytest
from langchain_core.documents import Document


class TestVectorStoreOps:
    """Test vector store CRUD.  Requires OPENAI_API_KEY for embeddings."""

    @pytest.fixture(autouse=True)
    def _patch_settings(self, monkeypatch, tmp_dir):
        """Point ChromaDB to a temp dir so tests don't touch real data."""
        monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_dir / "chroma"))
        monkeypatch.setattr("app.config.settings.chroma_collection_name", "test_collection")

    def test_get_stats_empty(self):
        from app.retrieval.vector_store import get_stats

        stats = get_stats()
        assert stats["count"] == 0

    @pytest.mark.skipif(
        not __import__("os").environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_add_and_query(self, tmp_dir):
        from app.retrieval.vector_store import add_documents, get_stats, query

        docs = [
            Document(page_content="Python is a programming language", metadata={"source": "test"}),
            Document(page_content="LangChain helps build LLM apps", metadata={"source": "test"}),
        ]
        add_documents(docs)
        stats = get_stats()
        assert stats["count"] == 2

        results = query("programming language", top_k=1)
        assert len(results) >= 1
        assert "Python" in results[0].page_content

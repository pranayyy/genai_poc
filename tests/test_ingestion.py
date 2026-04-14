"""Tests for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.preprocessor import preprocess


class TestLoaders:
    def test_load_json(self, sample_json_path: Path):
        docs = load_documents(str(sample_json_path), source_type="json")
        assert len(docs) == 2
        assert "Alice" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "json"
        assert "ingested_at" in docs[0].metadata

    def test_load_csv(self, sample_csv_path: Path):
        docs = load_documents(str(sample_csv_path), source_type="csv")
        assert len(docs) >= 2
        assert docs[0].metadata["source_type"] == "csv"

    def test_infer_type_json(self, sample_json_path: Path):
        docs = load_documents(str(sample_json_path))
        assert len(docs) == 2

    def test_unsupported_type_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unsupported"):
            load_documents("file.xyz", source_type="xyz")

    def test_infer_type_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Cannot infer"):
            load_documents("somefile.zzz")


class TestPreprocessor:
    def test_strips_whitespace(self):
        doc = Document(page_content="  hello   world  \n\n\n\n\nfoo  ")
        result = preprocess([doc])
        assert len(result) == 1
        assert "hello world" in result[0].page_content
        # Collapsed newlines
        assert "\n\n\n" not in result[0].page_content

    def test_removes_empty_docs(self):
        doc = Document(page_content="   \n\n  ")
        result = preprocess([doc])
        assert len(result) == 0

    def test_strips_page_numbers(self):
        doc = Document(page_content="Some text\nPage 3 of 10\nMore text")
        result = preprocess([doc])
        assert "Page 3 of 10" not in result[0].page_content


class TestChunker:
    def test_chunk_unstructured(self):
        text = "word " * 500  # ~2500 chars
        doc = Document(page_content=text, metadata={"source": "test", "source_type": "html"})
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # Each chunk has metadata
        assert all("chunk_id" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_structured_atomic(self, sample_json_path: Path):
        docs = load_documents(str(sample_json_path), source_type="json")
        chunks = chunk_documents(docs)
        # Structured data: each row stays as one chunk
        assert len(chunks) == 2
        assert "Fields:" in chunks[0].page_content

    def test_metadata_preserved(self):
        doc = Document(
            page_content="Test content " * 10,
            metadata={"source": "myfile.html", "source_type": "html", "custom_key": "val"},
        )
        chunks = chunk_documents([doc])
        assert chunks[0].metadata["source"] == "myfile.html"
        assert chunks[0].metadata["custom_key"] == "val"

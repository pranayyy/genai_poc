"""Tests for generation module."""

from __future__ import annotations

from app.generation.prompts import format_context


class TestPrompts:
    def test_format_context_basic(self):
        chunks = [
            {"content": "First chunk", "metadata": {"source": "doc1.pdf", "page": 1}},
            {"content": "Second chunk", "metadata": {"source": "doc2.html"}},
        ]
        result = format_context(chunks)
        assert "[1] Source: doc1.pdf, page 1" in result
        assert "[2] Source: doc2.html" in result
        assert "First chunk" in result
        assert "Second chunk" in result

    def test_format_context_empty(self):
        assert format_context([]) == ""

    def test_format_context_no_metadata(self):
        chunks = [{"content": "Some text", "metadata": {}}]
        result = format_context(chunks)
        assert "Source: unknown" in result

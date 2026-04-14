"""Embedding helper — delegates to the provider selected in settings."""

from __future__ import annotations

from app.llm_factory import get_embeddings

__all__ = ["get_embeddings"]

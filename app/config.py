"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM provider: "openai" or "groq" ───────────────────
    llm_provider: str = "groq"

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dims: int = 1536

    # ── Groq (free tier) ────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── HuggingFace local embeddings (free, no API key) ─────
    hf_embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "data/processed/chroma_db"
    chroma_collection_name: str = "faq_knowledge"

    # ── Retrieval ───────────────────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rerank_score_threshold: float = 0.3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Generation ──────────────────────────────────────────
    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # ── Guardrails ──────────────────────────────────────────
    offtopic_similarity_threshold: float = 0.30

    # ── Observability ───────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit: str = "10/minute"

    # ── Embedding batch ─────────────────────────────────────
    embedding_batch_size: int = 100


settings = Settings()

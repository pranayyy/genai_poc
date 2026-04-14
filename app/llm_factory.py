"""Provider-agnostic LLM and embedding factory.

Set LLM_PROVIDER=groq  (free, default) or LLM_PROVIDER=openai in .env.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel


@lru_cache(maxsize=1)
def get_llm() -> "BaseChatModel":
    """Return a configured chat model based on LLM_PROVIDER setting."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        )


@lru_cache(maxsize=1)
def get_embeddings() -> "Embeddings":
    """Return embeddings model.

    Uses HuggingFace local embeddings when provider is groq (free, no API key).
    Uses OpenAI embeddings when provider is openai.
    """
    if settings.llm_provider == "groq":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            # Model is already cached locally — skip HuggingFace Hub network checks.
            # Remove this flag only if you need to pull a model update.
            model_kwargs={"local_files_only": True},
        )
    else:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            dimensions=settings.openai_embedding_dims,
        )

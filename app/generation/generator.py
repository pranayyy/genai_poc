"""Answer generation with citation enforcement — works with any LangChain chat model."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.generation.prompts import SYSTEM_PROMPT, USER_PROMPT, format_context
from app.llm_factory import get_llm
from app.retrieval.retriever import RetrievedChunk


@dataclass
class SourceCitation:
    document: str
    chunk_id: str
    page: int | None = None
    relevance_score: float = 0.0


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    confidence: float = 1.0
    usage: dict | None = None


def generate(
    query: str,
    chunks: list[RetrievedChunk],
) -> GeneratedAnswer:
    """Generate a cited answer from retrieved *chunks*."""
    if not chunks:
        return GeneratedAnswer(
            answer="I don't have enough information to answer this question based on the available sources.",
            confidence=0.0,
        )

    # Build context
    chunk_dicts = [{"content": c.content, "metadata": c.metadata} for c in chunks]
    context_block = format_context(chunk_dicts)

    # Determine confidence
    low_conf = any(c.metadata.get("low_confidence") for c in chunks)
    confidence = 0.4 if low_conf else 1.0

    # Call LLM via factory (Groq or OpenAI)
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=USER_PROMPT.format(context=context_block, question=query)),
    ]
    response = llm.invoke(messages)
    answer_text = response.content or ""

    # Build source citations
    sources = [
        SourceCitation(
            document=c.metadata.get("source", "unknown"),
            chunk_id=c.metadata.get("chunk_id", ""),
            page=c.metadata.get("page", c.metadata.get("page_number")),
            relevance_score=c.score,
        )
        for c in chunks
    ]

    # Token usage (available on most providers via usage_metadata)
    usage = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        u = response.usage_metadata
        usage = {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }

    return GeneratedAnswer(
        answer=answer_text,
        sources=sources,
        confidence=confidence,
        usage=usage,
    )

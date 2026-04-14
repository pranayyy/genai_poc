"""Prompt templates for the generation stage."""

SYSTEM_PROMPT = """\
You are a knowledgeable FAQ assistant.  Your job is to answer the user's \
question using ONLY the context provided below.

RULES:
1. Base your answer strictly on the provided context.  Do NOT use outside knowledge.
2. Cite your sources inline using the format: [Source: <document>, page <page>] \
   or [Source: <document>] when no page is available.
3. If the context does not contain enough information to answer, respond with: \
   "I don't have enough information to answer this question based on the available sources."
4. Be concise but thorough.  Use bullet points when listing multiple items.
5. Never fabricate sources or facts.
"""

USER_PROMPT = """\
CONTEXT:
{context}

QUESTION:
{question}

Provide a well-structured answer with source citations.
"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into the context block for the prompt.

    Each *chunk* dict should have ``content`` and ``metadata`` keys.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_number"))
        label = f"[{i}] Source: {source}"
        if page is not None:
            label += f", page {page}"
        parts.append(f"{label}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

"""Output guardrails: faithfulness check and PII detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from app.llm_factory import get_llm


@dataclass
class OutputGuardResult:
    is_safe: bool
    warnings: list[str] = field(default_factory=list)


# ── PII regex patterns ────────────────────────────────────
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def check_output(
    answer: str,
    context_texts: list[str],
    check_faithfulness: bool = True,
) -> OutputGuardResult:
    """Run output safety checks on the generated *answer*."""
    warnings: list[str] = []

    # 1. PII scan
    for pii_type, pattern in _PII_PATTERNS.items():
        if pattern.search(answer):
            warnings.append(f"Potential PII detected: {pii_type}")

    # 2. Faithfulness check (LLM-based)
    if check_faithfulness and context_texts:
        faithful = _check_faithfulness(answer, context_texts)
        if not faithful:
            warnings.append("Answer may contain claims not supported by the retrieved context")

    is_safe = len(warnings) == 0
    return OutputGuardResult(is_safe=is_safe, warnings=warnings)


_FAITHFULNESS_PROMPT = """\
You are a fact-checking assistant.  Given retrieved CONTEXT and an ANSWER, \
determine whether EVERY claim in the ANSWER is supported by the CONTEXT.

Respond with ONLY "FAITHFUL" or "UNFAITHFUL".

CONTEXT:
{context}

ANSWER:
{answer}
"""


def _check_faithfulness(answer: str, context_texts: list[str]) -> bool:
    """Return True if every claim in *answer* is grounded in *context_texts*."""
    try:
        context = "\n---\n".join(context_texts[:5])  # limit tokens
        llm = get_llm()
        response = llm.invoke([
            HumanMessage(content=_FAITHFULNESS_PROMPT.format(context=context, answer=answer))
        ])
        verdict = (response.content or "").strip().upper()
        return verdict == "FAITHFUL"
    except Exception:
        # On failure, assume faithful to avoid wrongly blocking a good answer.
        return True

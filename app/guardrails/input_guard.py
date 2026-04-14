"""Input safety guardrails: injection, toxicity, off-topic detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Prompt-injection patterns ───────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
        r"(disregard|forget)\s+(all\s+)?(previous|above|prior)",
        r"you\s+are\s+now\s+(a|an|the)\b",
        r"system\s*prompt",
        r"reveal\s+(your|the)\s+(instructions|prompt|system)",
        r"act\s+as\s+(a|an|if)\b",
        r"pretend\s+(you|to)\b",
        r"do\s+not\s+follow\s+(your|the)\s+rules",
        r"new\s+instructions?\s*:",
        r"<\s*/?\s*system\s*>",
    ]
]


@dataclass
class GuardResult:
    is_safe: bool
    reason: str = ""
    category: str = ""


def check_input(query: str) -> GuardResult:
    """Run all input safety checks.  Returns on the first failure."""
    # 1. Basic length / emptiness
    if not query or not query.strip():
        return GuardResult(is_safe=False, reason="Empty query", category="invalid")

    if len(query) > 5000:
        return GuardResult(is_safe=False, reason="Query too long (max 5000 chars)", category="invalid")

    # 2. Prompt-injection regex
    for pat in _INJECTION_PATTERNS:
        if pat.search(query):
            return GuardResult(
                is_safe=False,
                reason="Potential prompt injection detected",
                category="injection",
            )

    # 3. OpenAI Moderation API (toxicity / harmful content)
    toxicity = _check_moderation(query)
    if toxicity is not None:
        return toxicity

    return GuardResult(is_safe=True)


def _check_moderation(query: str) -> GuardResult | None:
    """Call OpenAI Moderation endpoint (only when provider=openai and key is set).
    Returns a GuardResult on failure, else None.
    """
    from app.config import settings

    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return None  # skip when using Groq or no key configured

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.moderations.create(input=query)
        result = response.results[0]
        if result.flagged:
            flagged_cats = [
                cat for cat, flagged in result.categories.model_dump().items() if flagged
            ]
            return GuardResult(
                is_safe=False,
                reason=f"Content flagged: {', '.join(flagged_cats)}",
                category="toxicity",
            )
    except Exception:
        pass
    return None

"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval.retriever import RetrievedChunk


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """Re-score *chunks* with a cross-encoder and return the top results.

    If no chunk exceeds *score_threshold* the top 3 are returned with a
    ``low_confidence`` flag in metadata.
    """
    if not chunks:
        return []

    top_k = top_k or settings.rerank_top_k
    score_threshold = score_threshold if score_threshold is not None else settings.rerank_score_threshold

    model = _get_cross_encoder()
    pairs = [[query, c.content] for c in chunks]
    scores = model.predict(pairs).tolist()

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)

    above = [c for c in ranked if c.score >= score_threshold]
    if above:
        return above[:top_k]

    # Fallback: return top-3 with low-confidence flag
    fallback = ranked[:3]
    for c in fallback:
        c.metadata["low_confidence"] = True
    return fallback

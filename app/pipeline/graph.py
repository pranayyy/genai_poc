"""LangGraph RAG pipeline with guardrails and observability."""

from __future__ import annotations

import contextvars
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.generation.generator import GeneratedAnswer, generate
from app.guardrails.input_guard import GuardResult, check_input
from app.guardrails.output_guard import check_output
from app.observability.tracing import TraceContext
from app.retrieval.reranker import rerank
from app.retrieval.retriever import RetrievedChunk, retrieve

# ── ContextVar holds the TraceContext for the current pipeline run.
# LangGraph strips non-declared keys from state, so we use contextvars instead.
_current_trace: contextvars.ContextVar[TraceContext] = contextvars.ContextVar(
    "_current_trace", default=None  # type: ignore[arg-type]
)


def _trace() -> TraceContext:
    """Return the active TraceContext, creating a no-op fallback if missing."""
    t = _current_trace.get()
    if t is None:
        t = TraceContext()
    return t


# ── State schema ────────────────────────────────────────────


class PipelineState(TypedDict, total=False):
    query: str
    filters: dict[str, Any]
    guard_result: dict
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    generated_answer: dict
    output_guard_warnings: list[str]
    trace: dict
    error: str | None


# ── Node functions ──────────────────────────────────────────


def input_guard_node(state: PipelineState) -> PipelineState:
    trace = _trace()
    query = state["query"]

    with trace.stage("input_guard", input_size=len(query)) as rec:
        result: GuardResult = check_input(query)
        rec.output_size = 1 if result.is_safe else 0

    state["guard_result"] = {"is_safe": result.is_safe, "reason": result.reason, "category": result.category}
    return state


def retrieve_node(state: PipelineState) -> PipelineState:
    trace = _trace()
    query = state["query"]
    filters = state.get("filters", {})

    with trace.stage("retrieve", input_size=len(query)) as rec:
        chunks: list[RetrievedChunk] = retrieve(
            query,
            source_type=filters.get("source_type"),
            source=filters.get("source"),
        )
        rec.output_size = len(chunks)

    state["retrieved_chunks"] = [asdict(c) for c in chunks]
    return state


def rerank_node(state: PipelineState) -> PipelineState:
    trace = _trace()
    query = state["query"]
    chunks_raw = state.get("retrieved_chunks", [])

    chunks = [
        RetrievedChunk(content=c["content"], metadata=c["metadata"], score=c["score"])
        for c in chunks_raw
    ]

    with trace.stage("rerank", input_size=len(chunks)) as rec:
        reranked = rerank(query, chunks)
        rec.output_size = len(reranked)

    state["reranked_chunks"] = [asdict(c) for c in reranked]
    return state


def generate_node(state: PipelineState) -> PipelineState:
    trace = _trace()
    query = state["query"]
    chunks_raw = state.get("reranked_chunks", [])

    chunks = [
        RetrievedChunk(content=c["content"], metadata=c["metadata"], score=c["score"])
        for c in chunks_raw
    ]

    with trace.stage("generate", input_size=len(chunks)) as rec:
        answer: GeneratedAnswer = generate(query, chunks)
        rec.output_size = len(answer.answer)
        if answer.usage:
            rec.tokens = answer.usage.get("total_tokens", 0)

    state["generated_answer"] = {
        "answer": answer.answer,
        "sources": [asdict(s) for s in answer.sources],
        "confidence": answer.confidence,
        "usage": answer.usage,
    }
    return state


def output_guard_node(state: PipelineState) -> PipelineState:
    trace = _trace()
    answer_text = state.get("generated_answer", {}).get("answer", "")
    context_texts = [c["content"] for c in state.get("reranked_chunks", [])]

    with trace.stage("output_guard", input_size=len(answer_text)) as rec:
        result = check_output(answer_text, context_texts)
        rec.output_size = len(result.warnings)

    state["output_guard_warnings"] = result.warnings

    # If unfaithful, prepend a disclaimer
    if result.warnings:
        disclaimer = (
            "⚠️ **Note**: This answer may contain information not fully supported "
            "by the retrieved sources. Please verify independently.\n\n"
        )
        state["generated_answer"]["answer"] = disclaimer + state["generated_answer"]["answer"]

    return state


def blocked_response_node(state: PipelineState) -> PipelineState:
    guard = state.get("guard_result", {})
    state["generated_answer"] = {
        "answer": f"I'm unable to process this query. Reason: {guard.get('reason', 'blocked')}",
        "sources": [],
        "confidence": 0.0,
        "usage": None,
    }
    state["output_guard_warnings"] = []
    return state


def no_context_response_node(state: PipelineState) -> PipelineState:
    state["generated_answer"] = {
        "answer": "I don't have enough information to answer this question based on the available sources.",
        "sources": [],
        "confidence": 0.0,
        "usage": None,
    }
    state["output_guard_warnings"] = []
    return state


# ── Routing functions ───────────────────────────────────────


def route_after_guard(state: PipelineState) -> str:
    if state.get("guard_result", {}).get("is_safe"):
        return "retrieve"
    return "blocked_response"


def route_after_rerank(state: PipelineState) -> str:
    chunks = state.get("reranked_chunks", [])
    if chunks:
        return "generate"
    return "no_context_response"


# ── Graph construction ──────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("blocked_response", blocked_response_node)
    graph.add_node("no_context_response", no_context_response_node)

    # Set entry point
    graph.set_entry_point("input_guard")

    # Conditional edges
    graph.add_conditional_edges("input_guard", route_after_guard, {"retrieve": "retrieve", "blocked_response": "blocked_response"})
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", route_after_rerank, {"generate": "generate", "no_context_response": "no_context_response"})
    graph.add_edge("generate", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("blocked_response", END)
    graph.add_edge("no_context_response", END)

    return graph.compile()


# ── Convenience runner ──────────────────────────────────────


def run_pipeline(query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute the full RAG pipeline and return the result with trace info."""
    trace = TraceContext()
    # Set the trace in a ContextVar so nodes can access it without going
    # through LangGraph state (which strips undeclared keys).
    token = _current_trace.set(trace)
    try:
        pipeline = build_graph()

        initial_state: PipelineState = {
            "query": query,
            "filters": filters or {},
        }

        final_state = pipeline.invoke(initial_state)
    finally:
        _current_trace.reset(token)

    # Attach trace summary
    final_state["trace"] = trace.summary()

    return dict(final_state)

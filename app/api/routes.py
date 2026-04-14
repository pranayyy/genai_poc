"""API routes for the FAQ system."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel

from app.ingestion.chunker import chunk_documents
from app.ingestion.loaders import load_documents
from app.ingestion.preprocessor import preprocess
from app.observability.logger import get_logger
from app.pipeline.graph import run_pipeline
from app.retrieval import vector_store

router = APIRouter()
log = get_logger("routes")


# ── Request / Response models ───────────────────────────────


class QueryRequest(BaseModel):
    query: str
    filters: dict[str, Any] | None = None


class SourceCitationResponse(BaseModel):
    document: str
    chunk_id: str
    page: int | None = None
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitationResponse]
    confidence: float
    warnings: list[str]
    trace: dict[str, Any]


class IngestURLRequest(BaseModel):
    url: str
    source_type: str | None = None


class IngestResponse(BaseModel):
    message: str
    chunks_added: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    document_count: int


# ── Endpoints ───────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline for a user query."""
    log.info("query_received", query=body.query[:100])

    try:
        result = run_pipeline(body.query, filters=body.filters)
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid OpenAI API key. Update OPENAI_API_KEY in .env and restart the server.",
        )
    except RateLimitError as exc:
        detail = (
            "OpenAI quota exceeded. Add billing credits at "
            "https://platform.openai.com/settings/billing"
            if "insufficient_quota" in str(exc)
            else "OpenAI rate limit hit. Please retry in a moment."
        )
        raise HTTPException(status_code=503, detail=detail)
    except Exception as exc:
        log.error("pipeline_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    answer_data = result.get("generated_answer", {})
    return QueryResponse(
        answer=answer_data.get("answer", ""),
        sources=[SourceCitationResponse(**s) for s in answer_data.get("sources", [])],
        confidence=answer_data.get("confidence", 0.0),
        warnings=result.get("output_guard_warnings", []),
        trace=result.get("trace", {}),
    )


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(body: IngestURLRequest) -> IngestResponse:
    """Ingest a document from a URL."""
    log.info("ingest_url", url=body.url)
    try:
        docs = load_documents(body.url, source_type=body.source_type)
        docs = preprocess(docs)
        chunks = chunk_documents(docs)
        vector_store.add_documents(chunks)
        return IngestResponse(message="Ingestion complete", chunks_added=len(chunks))
    except RateLimitError:
        raise HTTPException(status_code=503, detail="OpenAI quota exceeded. Add billing credits at https://platform.openai.com/settings/billing")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
    except Exception as exc:
        log.error("ingest_failed", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_type: str | None = None,
) -> IngestResponse:
    """Ingest an uploaded file (PDF, HTML, CSV, JSON)."""
    log.info("ingest_file", filename=file.filename)

    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_documents(tmp_path, source_type=source_type)
        # Override source to use original filename
        for doc in docs:
            doc.metadata["source"] = file.filename or tmp_path
        docs = preprocess(docs)
        chunks = chunk_documents(docs)
        vector_store.add_documents(chunks)
        return IngestResponse(message="Ingestion complete", chunks_added=len(chunks))
    except RateLimitError:
        raise HTTPException(status_code=503, detail="OpenAI quota exceeded. Add billing credits at https://platform.openai.com/settings/billing")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
    except Exception as exc:
        log.error("ingest_failed", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return system health and collection stats."""
    stats = vector_store.get_stats()
    return HealthResponse(
        status="ok",
        collection=stats.get("collection", ""),
        document_count=stats.get("count", 0),
    )


@router.post("/evaluate")
async def evaluate_endpoint() -> dict:
    """Trigger an evaluation run using the golden dataset."""
    from app.evaluation.evaluator import run_evaluation

    log.info("evaluation_triggered")
    try:
        report = run_evaluation()
        return {"metrics": report.get("metrics", {}), "details": report.get("details", [])}
    except RateLimitError as exc:
        log.error("evaluation_rate_limited", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="LLM rate limit reached during evaluation. Wait a minute and retry, or increase EVAL_QUERY_DELAY in evaluator.py.",
        ) from exc
    except Exception as exc:
        log.error("evaluation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

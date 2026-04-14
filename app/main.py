"""FastAPI application entry point."""

from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.observability.logger import setup_logging

# Initialise logging before anything else
setup_logging()
log = structlog.get_logger("api")

# ── Rate limiter ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ── App ─────────────────────────────────────────────────────
app = FastAPI(title="GenAI FAQ", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_middleware(request: Request, call_next) -> Response:
    """Inject a trace_id into every request and log the request/response."""
    trace_id = uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    log.info("request_start", method=request.method, path=str(request.url.path))
    response: Response = await call_next(request)
    log.info("request_end", status=response.status_code)

    response.headers["X-Trace-Id"] = trace_id
    return response


# ── Register routes ─────────────────────────────────────────
from app.api.routes import router  # noqa: E402

app.include_router(router, prefix="/api")

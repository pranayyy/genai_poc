"""Request-level tracing: trace IDs, stage timing, token tracking."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import structlog


@dataclass
class StageRecord:
    name: str
    duration_ms: float = 0.0
    input_size: int = 0
    output_size: int = 0
    tokens: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """Per-request trace that records timing/stats for each pipeline stage."""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    stages: list[StageRecord] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, input_size: int = 0) -> Generator[StageRecord, None, None]:
        """Context manager that records a pipeline stage."""
        record = StageRecord(name=name, input_size=input_size)
        start = time.perf_counter()
        log = structlog.get_logger("tracing")
        try:
            yield record
        except Exception as exc:
            record.error = str(exc)
            raise
        finally:
            record.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.stages.append(record)
            log.info(
                "pipeline_stage",
                trace_id=self.trace_id,
                stage=name,
                duration_ms=record.duration_ms,
                input_size=record.input_size,
                output_size=record.output_size,
                tokens=record.tokens,
                error=record.error,
            )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all stages."""
        return {
            "trace_id": self.trace_id,
            "total_ms": round(sum(s.duration_ms for s in self.stages), 2),
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "input_size": s.input_size,
                    "output_size": s.output_size,
                    "tokens": s.tokens,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }

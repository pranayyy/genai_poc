"""Evaluation runner using RAGAS + custom metrics."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.evaluation.datasets import EvalSample, load_golden_dataset
from app.observability.logger import get_logger
from app.pipeline.graph import run_pipeline

log = get_logger("evaluator")


# Seconds to wait between pipeline calls to stay under Groq free-tier TPM (6,000).
# Groq free tier: 6,000 TPM. Each RAG query uses ~1,500-2,000 tokens.
# At 1 query/20s = 3 queries/min = ~5,400 tokens/min — safely under the limit.
# Set to 0 when using OpenAI or a paid Groq tier.
_EVAL_QUERY_DELAY: float = 40.0


def run_evaluation(
    golden_path: str = "eval/golden_qa.json",
    output_dir: str = "eval/results",
    query_delay: float = _EVAL_QUERY_DELAY,
) -> dict[str, Any]:
    """Run the full evaluation suite and return metrics + per-sample details.

    *query_delay* adds a sleep between each pipeline call to avoid hitting
    provider rate limits (Groq free tier: 6,000 TPM).
    """
    samples = load_golden_dataset(golden_path)
    log.info("evaluation_start", n_samples=len(samples), query_delay=query_delay)

    # Run each question through the pipeline
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for i, sample in enumerate(samples):
        if i > 0 and query_delay > 0:
            log.info("eval_rate_limit_delay", sample=i, delay_s=query_delay)
            time.sleep(query_delay)

        result = run_pipeline(sample.question)
        answer_data = result.get("generated_answer", {})

        questions.append(sample.question)
        answers.append(answer_data.get("answer", ""))
        contexts.append([c["content"] for c in result.get("reranked_chunks", [])])
        ground_truths.append(sample.ground_truth_answer)

    # Build RAGAS dataset
    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Run RAGAS evaluation
    ragas_result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    metrics: dict[str, float] = {k: float(v) for k, v in ragas_result.items() if isinstance(v, (int, float))}

    # Custom metrics
    citation_acc = _citation_accuracy(answers, contexts)
    refusal_acc = _refusal_accuracy(samples, answers)
    metrics["citation_accuracy"] = citation_acc
    metrics["refusal_accuracy"] = refusal_acc

    # Per-sample detail
    details = []
    for i, sample in enumerate(samples):
        details.append(
            {
                "question": sample.question,
                "answer": answers[i],
                "n_contexts": len(contexts[i]),
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(samples),
        "metrics": metrics,
        "details": details,
    }

    # Persist
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = out / f"eval_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("evaluation_complete", report=str(report_path), metrics=metrics)

    return report


# ── Custom metrics ──────────────────────────────────────────


def _citation_accuracy(answers: list[str], contexts: list[list[str]]) -> float:
    """Fraction of answers containing a [Source: ...] citation that actually references a retrieved context source."""
    citation_pat = re.compile(r"\[Source:\s*([^\]]+)\]")
    scored = 0
    total = 0
    for answer, ctx_list in zip(answers, contexts):
        citations = citation_pat.findall(answer)
        if not citations:
            continue
        total += 1
        ctx_text = " ".join(ctx_list).lower()
        if any(c.split(",")[0].strip().lower() in ctx_text for c in citations):
            scored += 1
    return scored / total if total > 0 else 1.0


def _refusal_accuracy(samples: list[EvalSample], answers: list[str]) -> float:
    """For samples with empty ground_truth_contexts (out-of-scope), check the system refused."""
    refusal_phrases = ["don't have enough information", "unable to", "cannot answer"]
    oos = [(s, a) for s, a in zip(samples, answers) if not s.ground_truth_contexts]
    if not oos:
        return 1.0
    correct = sum(
        1 for _, a in oos if any(phrase in a.lower() for phrase in refusal_phrases)
    )
    return correct / len(oos)

"""Golden evaluation dataset management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str] = field(default_factory=list)


def load_golden_dataset(path: str = "eval/golden_qa.json") -> list[EvalSample]:
    """Load evaluation Q/A pairs from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalSample(
            question=item["question"],
            ground_truth_answer=item["ground_truth_answer"],
            ground_truth_contexts=item.get("ground_truth_contexts", []),
        )
        for item in data
    ]

"""Run the evaluation suite from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.evaluator import run_evaluation


def main() -> None:
    golden_path = "eval/golden_qa.json"
    if not Path(golden_path).exists():
        print(f"Golden dataset not found at {golden_path}")
        return

    print("Running evaluation…")
    report = run_evaluation(golden_path)

    print("\n=== Evaluation Results ===")
    for metric, value in report.get("metrics", {}).items():
        print(f"  {metric:25s} {value:.4f}")
    print(f"\nFull report saved to eval/results/")


if __name__ == "__main__":
    main()

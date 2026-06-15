from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from src.multi_agent import RecommendationAgent, SupervisorAgent


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "data" / "multi_agent_eval_cases.csv"
DEFAULT_REPORT = ROOT / "reports" / "multi_agent_harness.csv"


def evaluate(
    cases_path: Path,
    full: bool = False,
    include_experiments: bool = False,
    pipeline_mode: str = "hybrid",
) -> pd.DataFrame:
    cases = pd.read_csv(cases_path)
    supervisor = SupervisorAgent(
        recommendation_agent=RecommendationAgent(pipeline_mode=pipeline_mode)
    )
    rows = []

    for _, case in cases.iterrows():
        # Each offline case is an isolated session so preferences cannot leak
        # from a previous test case. The retriever/model objects are reused.
        supervisor.reset()
        query = str(case["query"])
        expected = str(case["expected_intent"])
        predicted = supervisor.route(query)
        row = {
            "case_id": case["case_id"],
            "query": query,
            "expected_intent": expected,
            "predicted_intent": predicted,
            "route_correct": int(predicted == expected),
            "executed": 0,
            "task_success": None,
            "retry_count": None,
            "step_count": None,
            "latency_ms": None,
        }

        should_run = full and (predicted != "experiment" or include_experiments)
        if should_run:
            started = time.perf_counter()
            result = supervisor.run(query)
            elapsed_ms = (time.perf_counter() - started) * 1000
            critic = result.get("multi_agent_critic", {})
            row.update(
                {
                    "executed": 1,
                    "task_success": int(bool(critic.get("passed"))),
                    "retry_count": result.get("retry_count", 0),
                    "step_count": len(result.get("multi_agent_actions", [])),
                    "latency_ms": round(elapsed_ms, 2),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="FoodMate multi-agent evaluation harness")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--full", action="store_true", help="Execute agent tasks, not only routing")
    parser.add_argument(
        "--include-experiments",
        action="store_true",
        help="Also execute expensive offline experiment-agent cases",
    )
    parser.add_argument("--use-llm", action="store_true", help="Enable configured LLM calls during evaluation")
    parser.add_argument("--pipeline", default="hybrid", help="Recommendation pipeline used by full cases")
    args = parser.parse_args()

    if not args.use_llm:
        os.environ["FOODMATE_USE_LLM"] = "0"

    result = evaluate(args.cases, args.full, args.include_experiments, args.pipeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(result.to_string(index=False))
    print(f"\nRouting accuracy: {result['route_correct'].mean():.3f}")
    executed = result[result["executed"] == 1]
    if not executed.empty:
        print(f"Task success rate: {executed['task_success'].mean():.3f}")
        print(f"Average latency: {executed['latency_ms'].mean():.2f} ms")
        print(f"Average steps: {executed['step_count'].mean():.2f}")
        print(f"Average retries: {executed['retry_count'].mean():.2f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import evaluate


ROOT = Path(__file__).resolve().parent
A_B_PIPELINES = ["hybrid", "bge", "bge+cross_encoder", "bge+cross_encoder+business_rerank"]


def parse_args():
    parser = argparse.ArgumentParser(description="A/B evaluate retriever modes.")
    parser.add_argument("--data", default="data/restaurants_cuhksz.csv")
    parser.add_argument("--eval", default="data/eval_cases_cuhksz.csv")
    parser.add_argument("--out", default="reports/ab_evaluation_cuhksz.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for mode in A_B_PIPELINES:
        result = evaluate(eval_path=args.eval, data_path=args.data, retriever_mode=mode)
        metric_cols = [
            "hit_rate@5",
            "precision@5",
            "recall@5",
            "mrr@5",
            "ndcg@5",
            "budget_sat",
            "distance_sat",
            "cuisine_sat",
            "latency_ms",
        ]
        summary = result[metric_cols].mean(numeric_only=True).to_dict()
        summary["pipeline_mode"] = mode
        rows.append(summary)

    table = pd.DataFrame(rows)[
        [
            "pipeline_mode",
            "hit_rate@5",
            "precision@5",
            "recall@5",
            "mrr@5",
            "ndcg@5",
            "budget_sat",
            "distance_sat",
            "cuisine_sat",
            "latency_ms",
        ]
    ]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\nSaved A/B table to {out_path}")


if __name__ == "__main__":
    main()

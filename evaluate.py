from __future__ import annotations

import math
import argparse
import time
from pathlib import Path

import pandas as pd

from src.agent import FoodMateAgent, extract_preferences
from src.retriever import RestaurantRetriever
from src.utils import load_restaurants


ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT / "data" / "eval_cases.csv"
REPORT_PATH = ROOT / "reports" / "evaluation_results.csv"
PIPELINE_MODES = ["hybrid", "bge", "bge+cross_encoder", "bge+cross_encoder+business_rerank"]


def retriever_mode_for_pipeline(pipeline_mode: str) -> str:
    return "bge" if pipeline_mode.startswith("bge") else pipeline_mode


def parse_relevant(value: str) -> list[str]:
    return [item.strip() for item in str(value).split("|") if item.strip()]


def hit_rate_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    return float(any(name in relevant for name in recommended[:k]))


def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]).intersection(relevant)) / len(set(relevant))


def precision_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]).intersection(relevant)) / k


def mrr_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    rel_set = set(relevant)
    for idx, name in enumerate(recommended[:k], start=1):
        if name in rel_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    rel_set = set(relevant)
    dcg = 0.0
    for idx, name in enumerate(recommended[:k], start=1):
        if name in rel_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def approx_equal(a, b, tol=1e-6) -> bool:
    if pd.isna(a) or a == "":
        return True
    if b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def pref_match(row: pd.Series, prefs: dict, field: str) -> float:
    expected = row.get(f"expected_{field}")
    if pd.isna(expected) or expected == "":
        return 1.0
    actual = prefs.get(field)
    if field in ["budget", "max_distance_km"]:
        return float(approx_equal(expected, actual))
    if field in ["avoid_spicy", "vegetarian"]:
        return float(bool(int(expected)) == bool(actual))
    expected_values = [item.strip() for item in str(expected).split("|") if item.strip()]
    return float(str(actual) in expected_values)


def constraint_satisfaction(recs: pd.DataFrame, prefs: dict) -> dict:
    if recs.empty:
        return {"budget_sat": 0.0, "distance_sat": 0.0, "cuisine_sat": 0.0}
    budget = prefs.get("budget")
    distance = prefs.get("max_distance_km")
    cuisine = prefs.get("cuisine")
    budget_sat = 1.0 if not budget else float((recs["price_per_person"].astype(float) <= float(budget)).mean())
    distance_sat = 1.0 if not distance else float((recs["distance_km"].astype(float) <= float(distance)).mean())
    cuisine_sat = 1.0 if not cuisine else float((recs["cuisine_score"].astype(float) >= 0.9).mean())
    return {"budget_sat": budget_sat, "distance_sat": distance_sat, "cuisine_sat": cuisine_sat}


def evaluate(
    eval_path: str | Path = EVAL_PATH,
    data_path: str | Path | None = None,
    retriever_mode: str = "hybrid",
    case_limit: int | None = None,
) -> pd.DataFrame:
    cases = pd.read_csv(eval_path)
    if case_limit:
        cases = cases.head(int(case_limit))
    base_retriever_mode = retriever_mode_for_pipeline(retriever_mode)
    retriever = RestaurantRetriever(load_restaurants(data_path), mode=base_retriever_mode) if data_path else RestaurantRetriever(mode=base_retriever_mode)
    rows = []
    for _, case in cases.iterrows():
        agent = FoodMateAgent(retriever=retriever, pipeline_mode=retriever_mode)
        prefs = extract_preferences(case["query"])
        start = time.perf_counter()
        result = agent.handle(case["query"])
        latency_ms = (time.perf_counter() - start) * 1000
        recs = result["recommendations"]
        names = recs["name"].tolist() if not recs.empty else []
        relevant = parse_relevant(case["relevant_restaurants"])
        sat = constraint_satisfaction(recs, prefs)
        rows.append(
            {
                "case_id": case["case_id"],
                "hit_rate@5": hit_rate_at_k(names, relevant, 5),
                "precision@5": precision_at_k(names, relevant, 5),
                "recall@5": recall_at_k(names, relevant, 5),
                "mrr@5": mrr_at_k(names, relevant, 5),
                "ndcg@5": ndcg_at_k(names, relevant, 5),
                "budget_extract_acc": pref_match(case, prefs, "budget"),
                "cuisine_extract_acc": pref_match(case, prefs, "cuisine"),
                "scene_extract_acc": pref_match(case, prefs, "scene"),
                "distance_extract_acc": pref_match(case, prefs, "max_distance_km"),
                "avoid_spicy_extract_acc": pref_match(case, prefs, "avoid_spicy"),
                "vegetarian_extract_acc": pref_match(case, prefs, "vegetarian"),
                **sat,
                "latency_ms": latency_ms,
                "retriever_mode": base_retriever_mode,
                "pipeline_mode": retriever_mode,
                "recommended": "|".join(names),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FoodMate recommendation quality.")
    parser.add_argument("--eval", default=str(EVAL_PATH), help="Evaluation cases CSV.")
    parser.add_argument("--data", default=None, help="Restaurant data CSV.")
    parser.add_argument("--out", default=str(REPORT_PATH), help="Output evaluation CSV.")
    parser.add_argument("--retriever", default="hybrid", choices=["tfidf", "embedding", *PIPELINE_MODES], help="Retriever or pipeline mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(eval_path=args.eval, data_path=args.data, retriever_mode=args.retriever)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    metric_cols = [col for col in result.columns if col not in ["case_id", "recommended", "retriever_mode", "pipeline_mode"]]
    summary = result[metric_cols].mean(numeric_only=True).to_frame("mean").reset_index().rename(columns={"index": "metric"})
    print("Per-case results:")
    print(result.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()

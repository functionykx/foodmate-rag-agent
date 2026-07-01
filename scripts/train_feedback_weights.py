from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feedback import feedback_label, load_feedback
from src.reranker import DEFAULT_RERANK_WEIGHTS


FEATURE_TO_WEIGHT = {
    "semantic_score": "semantic",
    "cuisine_score": "cuisine",
    "budget_score": "budget",
    "distance_score": "distance",
    "travel_time_score": "travel_time",
    "rating_score": "rating",
    "scene_score": "scene",
    "deal_score": "deal",
    "spicy_score": "spicy",
    "feedback_boost": "feedback",
    "restaurant_quality_score": "restaurant_quality",
}


def _load_features(value: object) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, val in raw.items():
        try:
            result[key] = float(val)
        except (TypeError, ValueError):
            continue
    return result


def build_training_frame(feedback_path: Path) -> pd.DataFrame:
    feedback = load_feedback(feedback_path)
    rows = []
    for _, row in feedback.iterrows():
        label = feedback_label(row.get("feedback", ""), row.get("reason", ""))
        if label == 0:
            continue
        features = _load_features(row.get("features_json", ""))
        if not features:
            continue
        sample = {feature: features.get(feature, 0.0) for feature in FEATURE_TO_WEIGHT}
        sample["label"] = 1 if label > 0 else 0
        rows.append(sample)
    return pd.DataFrame(rows)


def train_weights(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or frame["label"].nunique() < 2:
        return dict(DEFAULT_RERANK_WEIGHTS)
    x = frame[list(FEATURE_TO_WEIGHT)].fillna(0.0)
    y = frame["label"].astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_scaled, y)
    coefs = pd.Series(model.coef_[0], index=x.columns)
    positive = coefs.clip(lower=0.0)
    if positive.sum() <= 1e-9:
        return dict(DEFAULT_RERANK_WEIGHTS)
    learned = positive / positive.sum()
    defaults = dict(DEFAULT_RERANK_WEIGHTS)
    total_default = sum(defaults.values())
    weights = {}
    for feature, weight_name in FEATURE_TO_WEIGHT.items():
        learned_weight = float(learned.get(feature, 0.0) * total_default)
        weights[weight_name] = round(0.7 * defaults[weight_name] + 0.3 * learned_weight, 4)
    return weights


def write_weights(path: Path, weights: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        lines = [f"{key}: {value}" for key, value in weights.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight rerank weights from FoodMate feedback.csv.")
    parser.add_argument("--feedback", default="reports/feedback.csv", help="Path to feedback.csv")
    parser.add_argument("--output", default="configs/rerank_weights.json", help="Output JSON path")
    args = parser.parse_args()

    frame = build_training_frame(Path(args.feedback))
    weights = train_weights(frame)
    output_path = Path(args.output)
    write_weights(output_path, weights)
    print(f"training_rows={len(frame)}")
    print(f"output={output_path}")
    print(json.dumps(weights, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

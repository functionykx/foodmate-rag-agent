from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


FEEDBACK_COLUMNS = [
    "timestamp",
    "user_id",
    "session_id",
    "domain",
    "item_id",
    "query",
    "restaurant_name",
    "item_name",
    "rank",
    "final_score",
    "feedback",
    "reason",
    "preferences_json",
    "features_json",
]

POSITIVE_FEEDBACK = {"like", "喜欢", "满意", "收藏", "good"}
NEGATIVE_FEEDBACK = {"dislike", "不喜欢", "不好", "bad"}
NEUTRAL_FEEDBACK = {"neutral", "一般", "还行", "看情况", "已去过"}
REASON_NEGATIVE = {"太贵", "太远", "不想吃这个菜系", "环境不合适", "不好吃"}


def log_feedback(
    query: str,
    restaurant_name: str,
    feedback: str,
    reason: str = "",
    user_id: str = "default_user",
    session_id: str = "",
    domain: str = "restaurant",
    item_id: str = "",
    item_name: str | None = None,
    rank: int | None = None,
    final_score: float | None = None,
    preferences: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    path: str | Path = "reports/feedback.csv",
    dedupe: bool = True,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display_name = item_name or restaurant_name
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "session_id": session_id,
        "domain": domain,
        "item_id": item_id,
        "query": query,
        "restaurant_name": restaurant_name,
        "item_name": display_name,
        "rank": rank,
        "final_score": final_score,
        "feedback": feedback,
        "reason": reason,
        "preferences_json": json.dumps(preferences or {}, ensure_ascii=False),
        "features_json": json.dumps(features or {}, ensure_ascii=False),
    }
    if dedupe and path.exists():
        existing = load_feedback(path)
        item_key = item_id or restaurant_name or display_name
        existing_item_key = existing["item_id"].fillna("")
        fallback_item_key = existing["restaurant_name"].fillna("").where(
            existing_item_key == "",
            existing_item_key,
        )
        duplicate = existing[
            (existing["user_id"].fillna("default_user") == user_id)
            & (existing["session_id"].fillna("") == session_id)
            & (existing["domain"].fillna("restaurant") == domain)
            & (fallback_item_key == item_key)
            & (existing["feedback"].fillna("") == feedback)
            & (existing["reason"].fillna("") == reason)
        ]
        if not duplicate.empty:
            saved = duplicate.iloc[-1].to_dict()
            saved["deduped"] = True
            return saved
    df = pd.DataFrame([row], columns=FEEDBACK_COLUMNS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return row


def load_feedback(path: str | Path = "reports/feedback.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    df = pd.read_csv(path)
    for column in FEEDBACK_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[FEEDBACK_COLUMNS]


def _load_json_cell(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}


def feedback_label(feedback: str, reason: str = "") -> int:
    text = str(feedback).strip()
    reason_text = str(reason).strip()
    if text in NEUTRAL_FEEDBACK or reason_text in NEUTRAL_FEEDBACK or any(token in text for token in ["一般", "还行", "已去过"]):
        return 0
    if text in POSITIVE_FEEDBACK or any(token in text for token in ["喜欢", "满意", "收藏", "好"]):
        return 1
    if text in NEGATIVE_FEEDBACK or reason_text in REASON_NEGATIVE or any(token in text for token in ["不喜欢", "不好", "差"]):
        return -1
    return 0


def build_user_profile(user_id: str = "default_user", path: str | Path = "reports/feedback.csv") -> dict[str, Any]:
    feedback = load_feedback(path)
    if feedback.empty:
        return empty_user_profile(user_id)
    scoped = feedback[(feedback["user_id"].fillna("default_user") == user_id) | (feedback["user_id"].fillna("") == "")]
    profile = empty_user_profile(user_id)
    for _, row in scoped.iterrows():
        label = feedback_label(row.get("feedback", ""), row.get("reason", ""))
        name = str(row.get("restaurant_name") or row.get("item_name") or "").strip()
        features = _load_json_cell(row.get("features_json", ""))
        cuisine = str(features.get("cuisine", "")).strip()
        reason = str(row.get("reason", "")).strip()
        if label > 0:
            if name:
                profile["liked_restaurants"][name] = profile["liked_restaurants"].get(name, 0) + 1
            if cuisine:
                profile["liked_cuisines"][cuisine] = profile["liked_cuisines"].get(cuisine, 0) + 1
        elif label < 0:
            if name:
                profile["disliked_restaurants"][name] = profile["disliked_restaurants"].get(name, 0) + 1
            if cuisine and reason == "不想吃这个菜系":
                profile["disliked_cuisines"][cuisine] = profile["disliked_cuisines"].get(cuisine, 0) + 1
            if reason == "太贵":
                profile["price_sensitivity"] += 0.1
            if reason == "太远":
                profile["distance_sensitivity"] += 0.1
            if reason:
                profile["avoid_reasons"].append(reason)
    profile["price_sensitivity"] = min(profile["price_sensitivity"], 1.0)
    profile["distance_sensitivity"] = min(profile["distance_sensitivity"], 1.0)
    profile["avoid_reasons"] = profile["avoid_reasons"][-10:]
    return profile


def empty_user_profile(user_id: str = "default_user") -> dict[str, Any]:
    return {
        "user_id": user_id,
        "liked_restaurants": {},
        "disliked_restaurants": {},
        "liked_cuisines": {},
        "disliked_cuisines": {},
        "price_sensitivity": 0.5,
        "distance_sensitivity": 0.5,
        "deal_sensitivity": 0.5,
        "avoid_reasons": [],
    }


def restaurant_quality_scores(path: str | Path = "reports/feedback.csv") -> dict[str, float]:
    feedback = load_feedback(path)
    if feedback.empty:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for _, row in feedback.iterrows():
        name = str(row.get("restaurant_name") or row.get("item_name") or "").strip()
        if not name:
            continue
        label = feedback_label(row.get("feedback", ""), row.get("reason", ""))
        bucket = stats.setdefault(name, {"positive": 0.0, "total": 0.0})
        if label > 0:
            bucket["positive"] += 1.0
            bucket["total"] += 1.0
        elif label < 0:
            bucket["total"] += 1.0
    # Bayesian smoothing keeps new restaurants close to neutral.
    alpha = 3.0
    beta = 3.0
    return {
        name: round((values["positive"] + alpha) / (values["total"] + alpha + beta), 4)
        for name, values in stats.items()
        if values["total"] > 0
    }

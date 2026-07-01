from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CUISINE_ALIASES = {
    "亚洲菜": ["亚洲菜", "中餐", "日料", "韩餐", "泰餐", "越南菜", "拉面", "寿司", "河粉", "饺子", "火锅"],
    "中餐": ["中餐", "中国菜", "家常菜", "饺子", "火锅", "粤菜", "湘菜", "川菜", "重庆火锅", "贵州菜", "客家菜", "潮汕菜", "浙菜", "徽菜", "江西菜", "西北菜", "牛肉面"],
    "粤菜": ["粤菜", "粤式", "早茶", "点心", "顺德菜", "茶餐厅", "海鲜"],
    "客家菜": ["客家菜", "客家", "酿豆腐", "盐焗鸡"],
    "潮汕菜": ["潮汕菜", "潮菜", "牛肉火锅", "粿条"],
    "火锅": ["火锅", "鸡煲", "涮锅"],
    "浙菜": ["浙菜", "江浙菜", "小海鲜"],
    "日料": ["日料", "寿司", "拉面", "便当"],
    "韩餐": ["韩餐", "泡菜", "部队锅"],
    "泰餐": ["泰餐", "冬阴功", "泰式"],
    "泰国菜": ["泰国菜", "泰餐", "泰式", "东南亚"],
    "越南菜": ["越南菜", "河粉", "檬粉"],
    "西餐": ["西餐", "披萨", "意面", "法餐", "牛排"],
    "轻食": ["轻食", "健康", "沙拉", "波奇饭", "低脂"],
    "甜品": ["甜品", "甜点", "蛋糕", "面包蛋糕", "面包", "奶茶", "双皮奶", "舒芙蕾", "漏奶华", "咖啡"],
    "咖啡": ["咖啡", "自习", "甜品"],
    "印度菜": ["印度菜", "咖喱"],
    "地中海菜": ["地中海菜", "卷饼", "希腊"],
}


SCENE_ALIASES = {
    "约会": ["约会", "浪漫", "纪念日", "生日", "安静"],
    "朋友聚餐": ["朋友", "聚餐", "多人", "大桌", "庆祝"],
    "快速午餐": ["快速", "出餐快", "午餐", "外带", "赶时间", "30 分钟"],
    "自习办公": ["自习", "办公", "学习", "安静", "插座"],
    "健康餐": ["健康", "健身", "低脂", "轻食", "低卡"],
    "独自用餐": ["一人食", "单人", "独自", "自己吃", "下午茶", "咖啡", "甜品", "小吃", "快餐"],
}


RELATED_CUISINES = {
    "中餐": ["粤菜", "湘菜", "客家菜", "潮汕菜", "浙菜", "徽菜", "贵州菜", "江西菜", "西北菜", "火锅"],
    "粤菜": ["顺德菜", "茶餐厅", "粤式茶点", "海鲜"],
    "潮汕菜": ["潮汕牛肉火锅", "牛肉火锅", "粿条"],
    "火锅": ["潮汕牛肉火锅", "鱼火锅", "云南火锅", "椰子鸡火锅", "重庆火锅", "本地鸡窝火锅"],
    "甜品": ["咖啡", "茶餐厅", "面包蛋糕", "小吃快餐"],
    "咖啡": ["甜品", "面包蛋糕", "茶餐厅"],
    "西餐": ["牛排", "意式餐厅", "西式快餐", "西班牙菜"],
    "韩餐": ["韩式料理", "韩式烤肉"],
    "泰国菜": ["泰餐", "东南亚菜"],
}


DEFAULT_RERANK_WEIGHTS = {
    "semantic": 0.35,
    "cuisine": 0.20,
    "budget": 0.15,
    "distance": 0.10,
    "travel_time": 0.10,
    "rating": 0.10,
    "scene": 0.10,
    "deal": 0.08,
    "spicy": 0.08,
    "feedback": 0.10,
    "restaurant_quality": 0.08,
}


def load_rerank_weights(path: str | Path | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_RERANK_WEIGHTS)
    configured = path or os.getenv("FOODMATE_RERANK_WEIGHTS_PATH")
    candidate_paths = [Path(configured)] if configured else [
        Path("configs/rerank_weights.yaml"),
        Path("configs/rerank_weights.yml"),
        Path("configs/rerank_weights.json"),
    ]
    weight_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if weight_path is None:
        return weights
    loaded = _load_weight_file(weight_path)
    for key, value in loaded.items():
        if key not in weights:
            continue
        try:
            weights[key] = float(value)
        except (TypeError, ValueError):
            continue
    return weights


def _load_weight_file(weight_path: Path) -> dict[str, float]:
    try:
        text = weight_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    text = text.lstrip("\ufeff")
    suffix = weight_path.suffix.lower()
    if suffix == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    if suffix in {".yaml", ".yml"}:
        return _parse_simple_yaml_mapping(text)
    return {}


def _parse_simple_yaml_mapping(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().strip("'\"")
        value = value.strip().strip("'\"")
        if not key:
            continue
        try:
            result[key] = float(value)
        except ValueError:
            continue
    return result


def _field_text(row: pd.Series) -> str:
    return " ".join(
        str(row[col]).lower()
        for col in ["cuisine", "tags", "best_for", "menu_highlights", "review_summary", "pros", "cons"]
    )


def _text_similarity(query: str, document: str) -> float:
    query = str(query).strip()
    document = str(document).strip()
    if not query or not document:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform([query, document])
        return float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    except ValueError:
        return 0.0


def cuisine_match(row: pd.Series, preferences: dict) -> float:
    cuisine = preferences.get("cuisine")
    if not cuisine:
        return 0.5
    text = _field_text(row)
    row_cuisine = str(row.get("cuisine", "")).lower()
    cuisine_text = str(cuisine).lower()
    if cuisine_text in row_cuisine:
        return 1.0
    exact_aliases = [alias.lower() for alias in CUISINE_ALIASES.get(cuisine, [cuisine])]
    if any(alias == row_cuisine or alias in row_cuisine for alias in exact_aliases):
        return 1.0
    related_aliases = [alias.lower() for alias in RELATED_CUISINES.get(cuisine, [])]
    if any(alias in row_cuisine or alias in text for alias in related_aliases):
        return 0.7
    if any(alias in text for alias in exact_aliases):
        return 0.7
    return 0.0


def scene_match(row: pd.Series, preferences: dict) -> float:
    scene = preferences.get("scene")
    if not scene:
        return 0.5
    text = _field_text(row)
    aliases = SCENE_ALIASES.get(scene, [scene])
    scene_query = " ".join([str(scene), *aliases])
    score = _text_similarity(scene_query, text)
    if any(alias.lower() in text for alias in aliases):
        score = max(score, 0.85)
    return round(min(max(score, 0.0), 1.0), 4)


def budget_match(row: pd.Series, preferences: dict) -> float:
    budget = preferences.get("budget")
    min_budget = preferences.get("min_budget")
    if not budget and not min_budget:
        return 0.5
    price = float(row["price_per_person"])
    if min_budget and budget:
        low = float(min_budget)
        high = float(budget)
        center = (low + high) / 2
        half_width = max((high - low) / 2, 1.0)
        distance = abs(price - center)
        return round(max(1.0 - distance / half_width, 0.0), 4)
    if min_budget:
        return 1.0 if price >= float(min_budget) else 0.0
    return 1.0 if price <= float(budget) else 0.0


def distance_match(row: pd.Series, preferences: dict) -> float:
    max_distance = preferences.get("max_distance_km")
    distance = float(row.get("effective_distance_km", row["distance_km"]))
    base_score = 1.0 / (1.0 + max(distance, 0.0))
    if not max_distance:
        return round(base_score, 4)
    if distance <= float(max_distance):
        return round(base_score, 4)
    return round(base_score * 0.5, 4)


def travel_time_match(row: pd.Series, preferences: dict) -> float:
    duration = row.get("estimated_duration_min")
    if duration is None or pd.isna(duration):
        return 0.5
    minutes = max(float(duration), 0.0)
    return round(1.0 / (1.0 + minutes / 10.0), 4)


def rating_score(row: pd.Series) -> float:
    return min(max((float(row["rating"]) - 3.5) / 1.5, 0.0), 1.0)


def dietary_penalty(row: pd.Series, preferences: dict) -> float:
    text = _field_text(row)
    penalties = 0.0
    if preferences.get("avoid_spicy") and ("辣" in text or "重口味" in text) and "不辣" not in text and "清淡" not in text:
        penalties += 0.2
    if preferences.get("wants_spicy") and spicy_match(row, preferences) < 0.9:
        penalties += 0.25
    if preferences.get("vegetarian") and "素食" not in text and "植物基" not in text:
        penalties += 0.25
    return penalties


def budget_range_penalty(row: pd.Series, preferences: dict) -> float:
    min_budget = preferences.get("min_budget")
    budget = preferences.get("budget")
    if not min_budget or not budget:
        return 0.0
    price = float(row["price_per_person"])
    low = float(min_budget)
    high = float(budget)
    if low <= price <= high:
        return 0.0
    if price < low:
        return 0.12 if price >= low * 0.8 else 0.25
    return 0.12 if price <= high * 1.15 else 0.25


def spicy_match(row: pd.Series, preferences: dict) -> float:
    if not preferences.get("wants_spicy"):
        return 0.5
    text = _field_text(row)
    keywords = ["辣", "麻辣", "香辣", "重口味", "湘菜", "川菜", "重庆火锅", "贵州菜", "酸汤", "辣椒炒肉", "剁椒"]
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0


def deal_match(row: pd.Series, preferences: dict) -> float:
    if not preferences.get("deal_preference"):
        return 0.5
    text = _field_text(row)
    keywords = ["优惠", "团购", "套餐", "工作日", "午市", "下午茶", "代金券"]
    return 1.0 if any(keyword in text for keyword in keywords) else 0.0


def rerank(candidates: pd.DataFrame, preferences: dict, top_k: int = 5) -> pd.DataFrame:
    candidates = candidates.copy()
    weights = load_rerank_weights()
    cuisine = preferences.get("cuisine")
    if cuisine:
        cuisine_pool = candidates[candidates.apply(lambda row: cuisine_match(row, preferences) >= 0.9, axis=1)]
        if not cuisine_pool.empty:
            candidates = cuisine_pool

    if preferences.get("wants_spicy"):
        spicy_pool = candidates[candidates.apply(lambda row: spicy_match(row, preferences) >= 0.9, axis=1)]
        if len(spicy_pool) >= top_k:
            candidates = spicy_pool

    budget = preferences.get("budget")
    min_budget = preferences.get("min_budget")
    if min_budget and budget:
        range_pool = candidates[
            (candidates["price_per_person"].astype(float) >= float(min_budget))
            & (candidates["price_per_person"].astype(float) <= float(budget))
        ]
        if len(range_pool) >= top_k:
            candidates = range_pool
    if budget:
        budget_pool = candidates[candidates["price_per_person"].astype(float) <= float(budget)]
        if len(budget_pool) >= top_k:
            candidates = budget_pool

    max_distance = preferences.get("max_distance_km")
    if max_distance:
        distance_column = "effective_distance_km" if "effective_distance_km" in candidates.columns else "distance_km"
        distance_pool = candidates[candidates[distance_column].astype(float) <= float(max_distance)]
        if len(distance_pool) >= top_k:
            candidates = distance_pool

    rows = []
    for _, row in candidates.iterrows():
        parts = {
            "semantic": float(row.get("semantic_similarity", 0.0)),
            "cuisine": cuisine_match(row, preferences),
            "budget": budget_match(row, preferences),
            "distance": distance_match(row, preferences),
            "travel_time": travel_time_match(row, preferences),
            "rating": rating_score(row),
            "scene": scene_match(row, preferences),
            "deal": deal_match(row, preferences),
            "spicy": spicy_match(row, preferences),
            "feedback": float(row.get("feedback_boost", 0.0)),
            "restaurant_quality": float(row.get("restaurant_quality_score", 0.5)),
        }
        score = (
            weights["semantic"] * parts["semantic"]
            + weights["cuisine"] * parts["cuisine"]
            + weights["budget"] * parts["budget"]
            + weights["distance"] * parts["distance"]
            + weights["travel_time"] * parts["travel_time"]
            + weights["rating"] * parts["rating"]
            + weights["scene"] * parts["scene"]
            + weights["deal"] * parts["deal"]
            + weights["spicy"] * parts["spicy"]
            + weights["feedback"] * parts["feedback"]
            + weights["restaurant_quality"] * parts["restaurant_quality"]
            - dietary_penalty(row, preferences)
            - budget_range_penalty(row, preferences)
        )
        scored = row.to_dict()
        scored.update({f"{key}_score": value for key, value in parts.items()})
        scored["final_score"] = max(score, 0.0)
        rows.append(scored)
    return pd.DataFrame(rows).sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

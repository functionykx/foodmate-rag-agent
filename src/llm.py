from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import pandas as pd


def llm_enabled() -> bool:
    return os.getenv("FOODMATE_USE_LLM", "").lower() in {"1", "true", "yes", "on"} and bool(os.getenv("OPENAI_API_KEY"))


class LLMClient:
    """Small OpenAI-compatible chat-completions client using only stdlib."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("FOODMATE_LLM_MODEL", "gpt-4o-mini")
        self.timeout = float(os.getenv("FOODMATE_LLM_TIMEOUT", "20"))

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str | None:
        if not self.api_key:
            return None
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError):
            return None


def _json_from_text(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def extract_preferences_with_llm(user_message: str, rule_preferences: dict) -> dict:
    client = LLMClient()
    prompt = f"""
你是餐厅推荐系统的需求理解模块。请从用户中文需求中抽取结构化偏好。

只输出 JSON，不要输出解释。字段可选：
- budget: 数字，人均预算，单位元
- min_budget: 数字，人均预算下限，单位元
- cuisine: 字符串，例如 粤菜、客家菜、火锅、咖啡、甜品、茶餐厅、西餐、湘菜、烤肉、轻食
- scene: 字符串，例如 朋友聚餐、快速午餐、约会、自习办公、独自用餐、夜宵、家庭聚餐
- max_distance_km: 数字，距离约束，单位公里
- avoid_spicy: 布尔值
- wants_spicy: 布尔值
- vegetarian: 布尔值
- deal_preference: 布尔值，用户是否在意团购、优惠、套餐、工作日午市、学生低预算
- user_location: 字符串，用户明确提供的当前位置或出发地点；没有明确提供时不要生成
- transport_mode: walking、riding 或 driving；用户没有说明时默认 walking

已有规则抽取结果：
{json.dumps(rule_preferences, ensure_ascii=False)}

用户需求：
{user_message}
"""
    text = client.chat(
        [
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    parsed = _json_from_text(text)
    merged = dict(rule_preferences)
    for key, value in parsed.items():
        if value not in [None, "", []]:
            merged[key] = value
    return merged


def generate_recommendation_text(user_message: str, preferences: dict, recommendations: pd.DataFrame) -> str | None:
    if recommendations.empty:
        return None

    client = LLMClient()
    rows = []
    for _, row in recommendations.iterrows():
        effective_distance = row.get("effective_distance_km", row["distance_km"])
        estimated_duration = row.get("estimated_duration_min")
        if pd.isna(estimated_duration):
            estimated_duration = None
        rows.append(
            {
                "name": row["name"],
                "cuisine": row["cuisine"],
                "price_per_person": row["price_per_person"],
                "rating": row["rating"],
                "distance_km": effective_distance,
                "distance_source": row.get("distance_source", "school_default"),
                "estimated_duration_min": estimated_duration,
                "transport_mode": row.get("transport_mode", preferences.get("transport_mode", "walking")),
                "final_score": round(float(row["final_score"]), 4),
                "budget_score": round(float(row["budget_score"]), 4),
                "distance_score": round(float(row["distance_score"]), 4),
                "cuisine_score": round(float(row["cuisine_score"]), 4),
                "scene_score": round(float(row["scene_score"]), 4),
                "deal_score": round(float(row.get("deal_score", 0.0)), 4),
                "menu_highlights": row["menu_highlights"],
                "review_summary": row["review_summary"],
                "pros": row["pros"],
                "cons": row["cons"],
            }
        )

    prompt = f"""
你是 FoodMate 餐厅推荐 Agent。请根据给定候选餐厅，生成中文 Top-5 推荐说明。

严格要求：
1. 只能使用候选餐厅 JSON 中的信息，不要编造菜单、地址、优惠或评价。
2. 五个推荐之间空一行。
3. 每家店只包含：店名、人均/距离/评分、推荐理由、可能缺点。
4. 每家店的推荐理由必须不超过 60 个汉字。
5. 只有 deal_score >= 0.9 且 JSON 文本明确出现团购/套餐/优惠时，才允许提优惠。
6. deal_score 为 0.5 表示中性未知，绝对不要写“有团购优惠”。
7. 不要输出字段名、JSON、评分表或“基于候选餐厅生成”等开场白。
8. 语气自然，适合项目 demo。

用户需求：
{user_message}

抽取偏好：
{json.dumps(preferences, ensure_ascii=False)}

候选餐厅：
{json.dumps(rows, ensure_ascii=False)}
"""
    return client.chat(
        [
            {"role": "system", "content": "你是严谨的可解释推荐系统助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

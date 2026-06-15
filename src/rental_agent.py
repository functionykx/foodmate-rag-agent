from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.neural_retrieval import CrossEncoderRanker
from src.tools.baidu_map import BaiduMapError, BaiduMapTool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENTAL_GEO_DATA = PROJECT_ROOT / "data" / "rental_listings_cuhksz_geo.csv"
DEFAULT_RENTAL_DATA = PROJECT_ROOT / "data" / "rental_listings_cuhksz.csv"


def resolve_rental_data_path() -> Path:
    configured = os.getenv("FOODMATE_RENTAL_DATA_PATH")
    if configured:
        return Path(configured)
    return DEFAULT_RENTAL_GEO_DATA if DEFAULT_RENTAL_GEO_DATA.exists() else DEFAULT_RENTAL_DATA


@dataclass
class RentalState:
    preferences: dict[str, Any] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    critic_report: dict[str, Any] = field(default_factory=dict)
    map_context: dict[str, Any] = field(default_factory=dict)


def _number(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def extract_rental_preferences(text: str) -> dict[str, Any]:
    """Extract practical rental constraints from a Chinese user query."""
    prefs: dict[str, Any] = {}
    normalized = str(text).replace(",", "，")

    occupant_match = re.search(r"(\d+|[一二两三四五六])\s*(?:人|个人)(?:一起)?(?:合租|住)", normalized)
    if occupant_match:
        raw_count = occupant_match.group(1)
        prefs["occupant_count"] = int(raw_count) if raw_count.isdigit() else {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        }[raw_count]

    per_person_match = re.search(
        r"(?:预算|租金|月租)?\s*(?:人均|每人)\s*(\d{3,5})\s*(?:元)?\s*(?:以内|以下|左右)?",
        normalized,
    )
    if per_person_match:
        prefs["per_person_rent"] = int(per_person_match.group(1))

    simple_budget_match = re.search(
        r"(?:预算|月租|租金)\s*(\d{3,5})\s*(?:元)?\s*(?:以内|以下|上限|封顶|左右)?\s*$",
        normalized,
    )
    if simple_budget_match and "per_person_rent" not in prefs:
        prefs["max_rent"] = int(simple_budget_match.group(1))

    range_match = re.search(r"(?:月租|租金|预算)?\s*(\d{3,5})\s*[-~到至]\s*(\d{3,5})", normalized)
    if range_match:
        low, high = sorted(map(int, range_match.groups()))
        prefs["min_rent"] = low
        prefs["max_rent"] = high
    else:
        max_match = re.search(r"(?:月租|租金|预算)?\s*(\d{3,5})\s*元?\s*(?:以内|以下|最多|封顶|左右)", normalized)
        if max_match:
            prefs["max_rent"] = int(max_match.group(1))
        elif any(token in normalized for token in ["租房", "房子", "房源", "公寓", "整租", "合租", "月租"]):
            rent_match = re.search(r"(?:月租|租金|预算)\s*(?:是|为|约|大概)?\s*(\d{3,5})", normalized)
            if rent_match:
                prefs["max_rent"] = int(rent_match.group(1))

    bedroom_patterns = [
        (r"(\d+)\s*室", "bedrooms"),
        (r"([一二两三四五])\s*室", "bedrooms_cn"),
        (r"([一二两三四五])\s*居", "bedrooms_cn"),
    ]
    for pattern, kind in bedroom_patterns:
        match = re.search(pattern, normalized)
        if match:
            is_minimum = bool(re.search(r"(?:至少|最少|不低于)\s*" + re.escape(match.group(0)), normalized))
            if kind == "bedrooms":
                bedroom_value = int(match.group(1))
            else:
                bedroom_value = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5}[match.group(1)]
            prefs["min_bedrooms" if is_minimum else "bedrooms"] = bedroom_value
            break
    if "开间" in normalized or "单间" in normalized:
        prefs.setdefault("bedrooms", 1)

    area_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:平|平方米|㎡)\s*(?:以上|起)", normalized)
    if area_match:
        prefs["min_area_sqm"] = float(area_match.group(1))

    orientations = ["东南", "西南", "南北", "东北", "西北", "朝南", "向南", "南", "东", "西", "北"]
    for orientation in orientations:
        if orientation in normalized:
            prefs["orientation"] = orientation.replace("朝", "").replace("向", "")
            break

    rental_type = None
    if "合租" in normalized:
        rental_type = "合租"
    elif "整租" in normalized or "独居" in normalized or "一个人住" in normalized:
        rental_type = "整租"
    if rental_type:
        prefs["rental_type"] = rental_type
    if prefs.get("occupant_count", 0) >= 2 and rental_type == "合租":
        # "两人合租" describes a household searching together, not
        # necessarily a platform listing whose rental type is shared-room.
        prefs["co_rent_group"] = True
        prefs.pop("rental_type", None)

    facilities = []
    facility_aliases = {
        "宽带": ["宽带", "wifi", "WiFi", "网络"],
        "天然气": ["天然气", "燃气", "可以做饭", "能做饭"],
        "空调": ["空调"],
        "冰箱": ["冰箱"],
        "洗衣机": ["洗衣机"],
        "电梯": ["电梯"],
        "停车": ["停车", "车位"],
    }
    for canonical, aliases in facility_aliases.items():
        if any(alias in normalized for alias in aliases):
            facilities.append(canonical)
    if facilities:
        prefs["required_facilities"] = facilities

    if "近地铁" in normalized or "地铁附近" in normalized or "离地铁近" in normalized:
        prefs["near_metro"] = True
    if "精装" in normalized or "拎包入住" in normalized:
        prefs["furnished"] = True
    if "随时入住" in normalized or "马上入住" in normalized:
        prefs["move_in"] = "随时入住"
    date_match = re.search(r"(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})", normalized)
    if date_match:
        prefs["move_in"] = date_match.group(1).replace("/", "-").replace(".", "-")
    if "短租" in normalized or "租期灵活" in normalized:
        prefs["short_lease"] = True
    if "验真" in normalized or "真实房源" in normalized or "不要广告" in normalized:
        prefs["verified_only"] = True

    destination_patterns = [
        r"(?:公司|单位|上班地点|工作地点|学校|通勤目的地)(?:在|是|位于|到)?[：:]?\s*([^，。；;]+)",
        r"(?:通勤到|每天去|上班去|到)\s*([^，。；;]+?)(?:通勤|上班|工作|，|。|；|;|$)",
    ]
    for pattern in destination_patterns:
        match = re.search(pattern, normalized)
        if match:
            destination = match.group(1).strip()
            if destination and not re.fullmatch(r"\d+分钟", destination):
                prefs["commute_destination"] = destination
                break
    if "港中深" in normalized or "香港中文大学（深圳）" in normalized or "香港中文大学深圳" in normalized:
        prefs["commute_destination"] = "香港中文大学（深圳）"
    if any(token in normalized for token in ["骑行", "骑车", "单车"]):
        prefs["transport_mode"] = "riding"
    elif any(token in normalized for token in ["驾车", "开车", "打车"]):
        prefs["transport_mode"] = "driving"
    elif any(token in normalized for token in ["步行", "走路"]):
        prefs["transport_mode"] = "walking"
    commute_match = re.search(r"(?:通勤|路上|车程|步行|骑行|驾车)?\s*(\d+)\s*分钟\s*(?:内|以内|以下|左右)?", normalized)
    if commute_match:
        prefs["max_commute_minutes"] = int(commute_match.group(1))
    return prefs


def merge_rental_preferences(current: dict[str, Any], new: dict[str, Any]) -> None:
    for key, value in new.items():
        if value not in (None, "", []):
            current[key] = value
    if "per_person_rent" in new:
        occupants = int(current.get("occupant_count", 1) or 1)
        current["max_rent"] = int(new["per_person_rent"]) * occupants
    if (
        int(current.get("occupant_count", 1) or 1) >= 2
        and (current.get("rental_type") == "合租" or current.get("co_rent_group"))
        and not current.get("bedrooms")
        and not current.get("min_bedrooms")
    ):
        current["preferred_min_bedrooms"] = int(current["occupant_count"])


def rental_followup(preferences: dict[str, Any]) -> str | None:
    if "max_rent" not in preferences:
        return "你的月租预算上限是多少？例如 2500 元、4000 元。"
    return None


class RentalRetriever:
    """Small hybrid retriever dedicated to the rental knowledge base."""

    def __init__(self, data_path: str | Path | None = None):
        self.data_path = Path(data_path) if data_path else resolve_rental_data_path()
        self.df = pd.read_csv(self.data_path)
        self.df["document"] = self.df.apply(self._document, axis=1)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["document"].fillna(""))
        n_docs, n_features = self.tfidf_matrix.shape
        components = max(1, min(64, n_docs - 1, n_features - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=42)
        self.dense_matrix = normalize(self.svd.fit_transform(self.tfidf_matrix))

    @staticmethod
    def _document(row: pd.Series) -> str:
        fields = [
            "title", "community", "district", "rental_type", "decoration", "orientation",
            "floor_level", "parking", "move_in_date", "lease_term", "viewing", "facilities",
            "tags", "verification_status", "publisher_compliance",
        ]
        content = " ".join(str(row.get(field, "")) for field in fields if pd.notna(row.get(field)))
        return f"{content} 月租{row.get('monthly_rent', '')}元 {row.get('bedrooms', '')}室 面积{row.get('area_sqm', '')}平方米"

    @staticmethod
    def _scale(values: np.ndarray) -> np.ndarray:
        low, high = float(values.min()), float(values.max())
        return np.zeros_like(values) if high - low < 1e-12 else (values - low) / (high - low)

    def search(self, query: str, top_k: int = 30) -> pd.DataFrame:
        query_vec = self.vectorizer.transform([query])
        sparse = cosine_similarity(query_vec, self.tfidf_matrix).ravel()
        dense_query = normalize(self.svd.transform(query_vec))
        dense = cosine_similarity(dense_query, self.dense_matrix).ravel()
        scores = 0.55 * self._scale(sparse) + 0.45 * self._scale(dense)
        idx = scores.argsort()[::-1][: min(top_k, len(self.df))]
        result = self.df.iloc[idx].copy()
        result["semantic_score"] = scores[idx]
        result["tfidf_score"] = sparse[idx]
        result["embedding_score"] = dense[idx]
        result["retriever_mode"] = "hybrid"
        return result.reset_index(drop=True)


def _rent_score(rent: float, preferences: dict[str, Any]) -> float:
    max_rent = preferences.get("max_rent")
    min_rent = preferences.get("min_rent")
    if max_rent is None:
        return 0.5
    if min_rent is not None:
        center = (float(min_rent) + float(max_rent)) / 2
        half_width = max((float(max_rent) - float(min_rent)) / 2, 300.0)
        return max(0.0, 1.0 - abs(rent - center) / (half_width * 2)) if rent <= max_rent * 1.1 else 0.0
    if rent <= max_rent:
        return max(0.65, 1.0 - abs(rent - max_rent * 0.85) / max(float(max_rent), 1.0))
    return max(0.0, 1.0 - (rent - max_rent) / max(float(max_rent) * 0.25, 500.0))


def rerank_rentals(candidates: pd.DataFrame, preferences: dict[str, Any], top_k: int = 5) -> pd.DataFrame:
    pool = candidates.copy()

    def narrow(mask: pd.Series) -> None:
        nonlocal pool
        filtered = pool[mask]
        if len(filtered) >= top_k:
            pool = filtered

    if preferences.get("max_rent"):
        budget_pool = pool[
            pd.to_numeric(pool["monthly_rent"], errors="coerce") <= float(preferences["max_rent"])
        ]
        if not budget_pool.empty:
            pool = budget_pool
    if preferences.get("bedrooms"):
        narrow(pd.to_numeric(pool["bedrooms"], errors="coerce") == int(preferences["bedrooms"]))
    if preferences.get("min_bedrooms"):
        narrow(pd.to_numeric(pool["bedrooms"], errors="coerce") >= int(preferences["min_bedrooms"]))
    if preferences.get("min_area_sqm"):
        narrow(pd.to_numeric(pool["area_sqm"], errors="coerce") >= float(preferences["min_area_sqm"]))
    if preferences.get("verified_only"):
        narrow(pool["verification_status"].astype(str).str.contains("贝壳验真", na=False))

    rows = []
    required = preferences.get("required_facilities", [])
    for _, row in pool.iterrows():
        rent = float(row.get("monthly_rent", 0) or 0)
        bedrooms = int(float(row.get("bedrooms", 0) or 0))
        area = float(row.get("area_sqm", 0) or 0)
        facilities_text = str(row.get("facilities", ""))
        tags = str(row.get("tags", ""))
        verification = str(row.get("verification_status", ""))

        budget_score = _rent_score(rent, preferences)
        preferred_min_bedrooms = preferences.get("preferred_min_bedrooms")
        target_min_bedrooms = preferences.get("min_bedrooms") or preferred_min_bedrooms
        if target_min_bedrooms:
            bedroom_score = 1.0 if bedrooms >= int(target_min_bedrooms) else max(0.0, 0.55 * bedrooms / int(target_min_bedrooms))
        else:
            bedroom_score = 1.0 if not preferences.get("bedrooms") else max(0.0, 1.0 - 0.5 * abs(bedrooms - int(preferences["bedrooms"])))
        area_score = 1.0 if not preferences.get("min_area_sqm") else min(1.0, area / float(preferences["min_area_sqm"]))
        facility_score = 0.5 if not required else sum(item in facilities_text or item == "电梯" and row.get("elevator") == "是" or item == "停车" and "车位" in str(row.get("parking", "")) for item in required) / len(required)
        metro_score = 1.0 if preferences.get("near_metro") and "近地铁" in tags else (0.0 if preferences.get("near_metro") else 0.5)
        furnished_score = 1.0 if preferences.get("furnished") and ("精装" in str(row.get("decoration", "")) or "精装" in tags) else (0.0 if preferences.get("furnished") else 0.5)
        orientation_score = 1.0 if preferences.get("orientation") and preferences["orientation"] in str(row.get("orientation", "")) else (0.0 if preferences.get("orientation") else 0.5)
        rental_type_score = 1.0 if preferences.get("rental_type") and preferences["rental_type"] in str(row.get("rental_type", "")) else (0.0 if preferences.get("rental_type") else 0.5)
        move_in_score = 0.5
        if preferences.get("move_in"):
            move_in_text = str(row.get("move_in_date", ""))
            move_in_score = 1.0 if preferences["move_in"] in move_in_text or "随时入住" in move_in_text else 0.25
        lease_score = 0.5
        if preferences.get("short_lease"):
            lease_text = f"{row.get('lease_term', '')} {tags}"
            lease_score = 1.0 if "灵活" in lease_text or "短租" in lease_text or "月租" in lease_text else 0.2
        verified_score = 1.0 if "贝壳验真" in verification else 0.35
        if preferences.get("verified_only") and verified_score < 1.0:
            verified_score = 0.0
        semantic = float(row.get("semantic_score", 0.0))
        duration = row.get("commute_duration_min")
        if duration is None or pd.isna(duration):
            commute_score = 0.5
        else:
            duration = float(duration)
            max_minutes = preferences.get("max_commute_minutes")
            if max_minutes:
                commute_score = max(0.0, 1.0 - duration / max(float(max_minutes) * 2, 1.0))
                if duration <= float(max_minutes):
                    commute_score = max(commute_score, 0.75)
            else:
                commute_score = 1.0 / (1.0 + duration / 20.0)
        final = (
            0.14 * semantic + 0.24 * budget_score + 0.18 * bedroom_score +
            0.06 * area_score + 0.06 * facility_score + 0.04 * metro_score +
            0.04 * furnished_score + 0.03 * orientation_score + 0.05 * verified_score +
            0.04 * rental_type_score + 0.03 * move_in_score + 0.03 * lease_score +
            0.13 * commute_score
        )
        enriched = row.to_dict()
        enriched.update({
            "final_score": final,
            "budget_score": budget_score,
            "bedroom_score": bedroom_score,
            "bedroom_preference_met": 1.0 if not preferred_min_bedrooms or bedrooms >= int(preferred_min_bedrooms) else 0.0,
            "area_score": area_score,
            "facility_score": facility_score,
            "metro_score": metro_score,
            "furnished_score": furnished_score,
            "orientation_score": orientation_score,
            "verified_score": verified_score,
            "rental_type_score": rental_type_score,
            "move_in_score": move_in_score,
            "lease_score": lease_score,
            "commute_score": commute_score,
        })
        rows.append(enriched)
    result = pd.DataFrame(rows)
    sort_columns = ["final_score"]
    if preferences.get("preferred_min_bedrooms"):
        sort_columns = ["bedroom_preference_met", "final_score"]
    return result.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(top_k).reset_index(drop=True)


def render_rentals(recommendations: pd.DataFrame, preferences: dict[str, Any]) -> str:
    if recommendations.empty:
        return "暂时没有找到符合条件的房源，请放宽预算或户型要求。"
    blocks = ["我根据你的租房需求筛选了这些房源："]
    preferred_min = preferences.get("preferred_min_bedrooms")
    if preferred_min and not recommendations.empty:
        matched = pd.to_numeric(recommendations["bedrooms"], errors="coerce") >= int(preferred_min)
        if not matched.any():
            blocks.append(
                f"当前预算内暂无 {preferred_min} 室房源，以下为预算内的一室备选；"
                "如需独立卧室，建议提高总预算。"
            )
    for index, row in recommendations.iterrows():
        reasons = []
        if row["budget_score"] >= 0.75:
            reasons.append("租金接近预算")
        if row["bedroom_score"] >= 0.9 and (
            preferences.get("bedrooms")
            or preferences.get("min_bedrooms")
            or preferences.get("preferred_min_bedrooms")
        ):
            reasons.append("户型匹配")
        if row["metro_score"] >= 0.9:
            reasons.append("近地铁")
        if row["verified_score"] >= 0.9:
            reasons.append("有贝壳验真编号")
        if pd.notna(row.get("commute_duration_min")):
            reasons.append(f"通勤约{float(row['commute_duration_min']):.0f}分钟")
        reason = "、".join(reasons[:3]) or "综合匹配度较高"
        facilities = str(row.get("facilities", "")).replace(";", "、") or "未提供"
        warning = "广告房源，签约前需重点核验" if row["verified_score"] < 0.9 else "费用及在租状态需再次确认"
        blocks.append(
            f"{index + 1}. {row['title']}\n"
            f"月租 {int(row['monthly_rent'])} 元，{int(row['bedrooms'])}室，{row['area_sqm']}㎡，{row['orientation']}。\n"
            f"推荐理由：{reason}。\n"
            f"设施：{facilities}。\n"
            f"提醒：{warning}。"
        )
    return "\n\n".join(blocks)


class RentalAgent:
    """Stateful RAG agent for rental search and constraint refinement."""

    def __init__(self, retriever: RentalRetriever | None = None, map_tool: BaiduMapTool | None = None):
        self.retriever = retriever or RentalRetriever()
        self.map_tool = map_tool or BaiduMapTool()
        self.pipeline_mode = os.getenv("FOODMATE_RENTAL_PIPELINE_MODE", "hybrid").lower()
        self.cross_encoder = CrossEncoderRanker() if "cross_encoder" in self.pipeline_mode else None
        self.state = RentalState()

    def reset(self) -> None:
        self.state = RentalState()

    def handle(self, user_query: str) -> dict[str, Any]:
        self.state.actions = []
        self.state.turns.append({"role": "user", "content": user_query})
        extracted = extract_rental_preferences(user_query)
        bare_rent = re.fullmatch(r"\s*(\d{3,5})\s*(?:元)?\s*", str(user_query))
        if bare_rent and "max_rent" not in self.state.preferences:
            extracted["max_rent"] = int(bare_rent.group(1))
        merge_rental_preferences(self.state.preferences, extracted)
        self.state.actions.append({"agent": "rental_agent", "action": "extract_preferences", "detail": extracted})
        self.state.plan = [
            "extract_rental_preferences", "hybrid_retrieval_top30", "semantic_rerank",
            "business_prerank_top10", "baidu_route_top10", "commute_rerank_top5", "critic", "answer",
        ]

        question = rental_followup(self.state.preferences)
        if question:
            return self._response("followup", question, pd.DataFrame())

        recall_k = int(os.getenv("FOODMATE_RECALL_TOP_K", "30"))
        query = self._rewrite_query(user_query)
        candidates = self.retriever.search(query, top_k=recall_k)
        self.state.candidates = candidates
        self.state.actions.append({"agent": "rental_agent", "action": "hybrid_retrieval", "detail": {"query": query, "top_k": recall_k, "candidates": len(candidates)}})
        if self.cross_encoder is not None:
            candidates = self.cross_encoder.rerank(query, candidates)
            candidates["semantic_score"] = candidates["semantic_similarity"].astype(float)
            self.state.actions.append({
                "agent": "rental_agent", "action": "cross_encoder_rerank",
                "detail": {"model": self.cross_encoder.model_name, "candidates": len(candidates)},
            })
        else:
            self.state.actions.append({
                "agent": "rental_agent", "action": "semantic_rerank_skipped",
                "detail": {"pipeline_mode": self.pipeline_mode},
            })

        preranked = rerank_rentals(candidates, self.state.preferences, top_k=10)
        self.state.actions.append({"agent": "rental_agent", "action": "business_prerank", "detail": {"top_k": len(preranked)}})
        routed = self._commute_route_tool(preranked)
        recommendations = rerank_rentals(routed, self.state.preferences, top_k=5)
        self.state.recommendations = recommendations
        self.state.actions.append({"agent": "rental_agent", "action": "commute_business_rerank", "detail": {"top_k": len(recommendations)}})
        self.state.critic_report = self._validate(recommendations)
        self.state.actions.append({"agent": "rental_agent", "action": "critic_validate", "detail": self.state.critic_report})
        return self._response("rental_recommendation", render_rentals(recommendations, self.state.preferences), recommendations)

    def _rewrite_query(self, user_query: str) -> str:
        parts = [user_query]
        prefs = self.state.preferences
        if prefs.get("max_rent"):
            parts.append(f"月租预算{prefs['max_rent']}元以内")
        if prefs.get("bedrooms"):
            parts.append(f"{prefs['bedrooms']}室")
        if prefs.get("min_bedrooms"):
            parts.append(f"至少{prefs['min_bedrooms']}室")
        elif prefs.get("preferred_min_bedrooms"):
            parts.append(f"优先{prefs['preferred_min_bedrooms']}室及以上")
        if prefs.get("required_facilities"):
            parts.extend(prefs["required_facilities"])
        if prefs.get("near_metro"):
            parts.append("近地铁")
        if prefs.get("commute_destination"):
            parts.append(f"通勤到{prefs['commute_destination']}")
        return " ".join(parts)

    def _commute_route_tool(self, candidates: pd.DataFrame) -> pd.DataFrame:
        result = candidates.copy()
        result["commute_distance_km"] = pd.NA
        result["commute_duration_min"] = pd.NA
        result["commute_source"] = "not_requested"
        destination = self.state.preferences.get("commute_destination")
        mode = self.state.preferences.get("transport_mode", "walking")
        if not destination:
            self.state.map_context = {"enabled": False, "reason": "commute_destination_not_provided"}
            self.state.actions.append({"agent": "rental_agent", "action": "baidu_route_skipped", "detail": self.state.map_context})
            return result
        if not self.map_tool.available():
            self.state.map_context = {"enabled": False, "reason": "BAIDU_MAP_AK_not_configured"}
            self.state.actions.append({"agent": "rental_agent", "action": "baidu_route_skipped", "detail": self.state.map_context})
            return result

        valid_indices = []
        origins = []
        for idx, row in result.iterrows():
            try:
                latitude = float(row.get("latitude"))
                longitude = float(row.get("longitude"))
            except (TypeError, ValueError):
                continue
            if pd.isna(latitude) or pd.isna(longitude):
                continue
            valid_indices.append(idx)
            origins.append((latitude, longitude))
        try:
            target = self.map_tool.resolve_location(str(destination))
            target_point = (float(target["latitude"]), float(target["longitude"]))
            routes = self.map_tool.route_many_origins_to_destination(
                origins, target_point, transport_mode=mode
            )
            for idx, route in zip(valid_indices, routes):
                result.at[idx, "commute_distance_km"] = route["distance_km"]
                result.at[idx, "commute_duration_min"] = route["duration_min"]
                result.at[idx, "commute_source"] = f"baidu_{mode}"
            self.state.map_context = {
                "enabled": True, "destination": destination, "transport_mode": mode,
                "destination_point": target, "routed_candidates": len(routes),
                "fallback_candidates": len(result) - len(routes),
            }
            self.state.actions.append({"agent": "rental_agent", "action": f"baidu_{mode}_route_top10", "detail": self.state.map_context})
        except BaiduMapError as exc:
            self.state.map_context = {"enabled": False, "reason": str(exc), "destination": destination}
            self.state.actions.append({"agent": "rental_agent", "action": "baidu_route_fallback", "detail": self.state.map_context})
        return result

    @staticmethod
    def _validate(recommendations: pd.DataFrame) -> dict[str, Any]:
        required = {"listing_id", "title", "monthly_rent", "final_score", "budget_score", "verified_score"}
        missing = sorted(required.difference(recommendations.columns)) if not recommendations.empty else sorted(required)
        return {"passed": not recommendations.empty and not missing, "issues": [] if not missing else [f"缺少字段: {missing}"], "top_k": len(recommendations)}

    def _response(self, result_type: str, message: str, recommendations: pd.DataFrame) -> dict[str, Any]:
        return {
            "type": result_type,
            "domain": "rental",
            "message": message,
            "preferences": dict(self.state.preferences),
            "plan": list(self.state.plan),
            "actions": list(self.state.actions),
            "recommendations": recommendations,
            "critic_report": dict(self.state.critic_report),
            "memory": {"turn_count": len(self.state.turns)},
            "map_context": dict(self.state.map_context),
        }

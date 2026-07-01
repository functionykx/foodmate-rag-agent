from __future__ import annotations

import re
import os
from dataclasses import dataclass, field

import pandas as pd

from src.prompts import build_query
from src.neural_retrieval import CrossEncoderRanker
from src.reranker import cuisine_match, rerank, spicy_match
from src.retriever import RestaurantRetriever
from src.utils import contains_any, normalize_text
from src.llm import extract_preferences_with_llm, generate_recommendation_text, llm_enabled
from src.feedback import build_user_profile, restaurant_quality_scores
from src.tools.baidu_map import BaiduMapError, BaiduMapTool


@dataclass
class AgentState:
    preferences: dict = field(default_factory=dict)
    turns: list[dict] = field(default_factory=list)
    intent: str = "new_recommendation"
    plan: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    critic_report: dict = field(default_factory=dict)
    final_query: str = ""
    map_context: dict = field(default_factory=dict)


class FoodMateAgent:
    def __init__(
        self,
        retriever: RestaurantRetriever | None = None,
        pipeline_mode: str | None = None,
        map_tool: BaiduMapTool | None = None,
    ):
        self.pipeline_mode = (pipeline_mode or os.getenv("FOODMATE_PIPELINE_MODE", "hybrid")).lower()
        retriever_mode = "bge" if self.pipeline_mode.startswith("bge") else None
        self.retriever = retriever or RestaurantRetriever(mode=retriever_mode)
        self.cross_encoder = CrossEncoderRanker() if "cross_encoder" in self.pipeline_mode else None
        self.map_tool = map_tool or BaiduMapTool()
        self.state = AgentState()
        self.pending_critic_feedback: dict = {}

    def reset(self) -> None:
        self.state = AgentState()
        self.pending_critic_feedback = {}

    def apply_critic_feedback(self, report: dict) -> None:
        """Persist supervisor feedback across the next turn reset."""
        self.pending_critic_feedback = dict(report or {})

    def handle(self, user_message: str) -> dict:
        self._start_turn(user_message)
        self._intent_router(user_message)
        self._preference_extractor(user_message)
        self._planner()
        self._memory_lookup(user_message)
        missing = self._clarification_node()

        if missing:
            question = ask_followup(missing[0])
            self._record_action("clarification", "ask_user", {"missing_field": missing[0], "question": question})
            return {
                "type": "followup",
                "message": question,
                "preferences": dict(self.state.preferences),
                "intent": self.state.intent,
                "plan": list(self.state.plan),
                "actions": list(self.state.actions),
                "memory": dict(self.state.memory),
                "critic_report": {},
                "recommendations": pd.DataFrame(),
            }

        self._retrieval_tool(user_message)
        self._cross_encoder_tool()
        self._map_distance_tool()
        self._ranking_tool()
        self._memory_rerank_tool()
        self._critic_validator()
        if not self.state.critic_report.get("passed", True):
            self._repair_tool()

        recommendations = self.state.recommendations
        message = None
        answer_source = "template"
        if llm_enabled():
            message = generate_recommendation_text(user_message, self.state.preferences, recommendations)
            if message:
                answer_source = "llm"
        if not message:
            message = render_recommendations(recommendations, self.state.preferences)
        if 0 < len(recommendations) < 5:
            message = (
                f"在菜系与预算上浮 15% 容差内只有 {len(recommendations)} 家，"
                "未使用其他菜系凑数。\n\n" + message
            )
        self._record_action("answer", "generate_response", {"source": answer_source})
        return {
            "type": "recommendation",
            "message": message,
            "preferences": dict(self.state.preferences),
            "intent": self.state.intent,
            "plan": list(self.state.plan),
            "actions": list(self.state.actions),
            "memory": dict(self.state.memory),
            "critic_report": dict(self.state.critic_report),
            "map_context": dict(self.state.map_context),
            "recommendations": recommendations,
        }

    def _start_turn(self, user_message: str) -> None:
        self.state.actions = []
        self.state.plan = []
        self.state.memory = {}
        self.state.candidates = pd.DataFrame()
        self.state.recommendations = pd.DataFrame()
        self.state.critic_report = {}
        self.state.final_query = ""
        self.state.map_context = {}
        self.state.turns.append({"role": "user", "content": user_message})

    def _record_action(self, node: str, action: str, detail: dict | None = None) -> None:
        self.state.actions.append({"node": node, "action": action, "detail": detail or {}})

    def _intent_router(self, user_message: str) -> None:
        text = normalize_text(user_message)
        if contains_any(text, ["为什么", "原因", "解释", "依据", "为什么推荐"]):
            intent = "explain_recommendation"
        elif contains_any(text, ["不喜欢", "太贵", "太远", "不好吃", "不想吃", "换一家", "不要"]):
            intent = "feedback_or_constraint_update"
        elif self.state.preferences and extract_preferences(user_message):
            intent = "constraint_update"
        else:
            intent = "new_recommendation"
        self.state.intent = intent
        self._record_action("intent_router", "classify_intent", {"intent": intent})

    def _preference_extractor(self, user_message: str) -> None:
        extracted = extract_preferences(user_message)
        source = "rules"
        if llm_enabled():
            extracted = extract_preferences_with_llm(user_message, extracted)
            source = "rules+llm"
        merge_preferences(self.state.preferences, extracted)
        self.state.turns[-1]["preferences"] = dict(self.state.preferences)
        self._record_action("preference_extractor", "extract_preferences", {"source": source, "extracted": extracted})

    def _planner(self) -> None:
        plan = ["memory_lookup", "clarification"]
        missing = missing_required_fields(self.state.preferences)
        if not missing:
            plan.extend(["query_rewrite", "retrieval"])
            if self.cross_encoder is not None:
                plan.append("cross_encoder_rerank")
            if self.state.preferences.get("user_location"):
                mode = self.state.preferences.get("transport_mode", "walking")
                plan.append(f"baidu_{mode}_route")
            if self.pipeline_mode.endswith("business_rerank") or self.pipeline_mode == "hybrid":
                plan.append("business_rerank")
            else:
                plan.append("semantic_topk")
            plan.extend(["memory_rerank", "critic_validator", "answer"])
        self.state.plan = plan
        self._record_action("planner", "build_plan", {"plan": plan})

    def _memory_lookup(self, user_message: str) -> None:
        user_id = os.getenv("FOODMATE_USER_ID", "default_user")
        profile = build_user_profile(user_id=user_id)
        quality_scores = restaurant_quality_scores()
        memory = {
            **profile,
            "restaurant_quality_scores": quality_scores,
            "liked": list(profile["liked_restaurants"].keys())[-10:],
            "disliked": list(profile["disliked_restaurants"].keys())[-10:],
        }
        self.state.memory = memory
        self._record_action(
            "memory",
            "load_feedback_memory",
            {
                "user_id": user_id,
                "liked": memory["liked"],
                "disliked": memory["disliked"],
                "liked_cuisines": profile["liked_cuisines"],
                "disliked_cuisines": profile["disliked_cuisines"],
                "price_sensitivity": profile["price_sensitivity"],
                "distance_sensitivity": profile["distance_sensitivity"],
                "restaurant_quality_count": len(quality_scores),
            },
        )

    def _clarification_node(self) -> list[str]:
        missing = missing_required_fields(self.state.preferences)
        self._record_action("clarification", "check_missing_fields", {"missing": missing})
        return missing

    def _retrieval_tool(self, user_message: str) -> None:
        query = build_query(self.state.preferences, user_message)
        self.state.final_query = query
        recall_top_k = int(os.getenv("FOODMATE_RECALL_TOP_K", "30"))
        candidates = self.retriever.search(query, top_k=recall_top_k)
        self.state.candidates = candidates
        self._record_action(
            "tool_router",
            "search_restaurants",
            {"query": query, "top_k": recall_top_k, "num_candidates": len(candidates)},
        )

    def _cross_encoder_tool(self) -> None:
        if self.cross_encoder is None or self.state.candidates.empty:
            return
        self.state.candidates = self.cross_encoder.rerank(self.state.final_query, self.state.candidates)
        self._record_action(
            "tool_router",
            "cross_encoder_rerank",
            {"model": self.cross_encoder.model_name, "num_candidates": len(self.state.candidates)},
        )

    def _map_distance_tool(self) -> None:
        user_location = self.state.preferences.get("user_location")
        transport_mode = self.state.preferences.get("transport_mode", "walking")
        if not user_location or self.state.candidates.empty:
            self.state.map_context = {
                "enabled": False,
                "distance_source": "school_default",
                "reason": "user_location_not_provided",
            }
            return
        if not self.map_tool.available():
            self.state.map_context = {
                "enabled": False,
                "distance_source": "school_fallback",
                "reason": "BAIDU_MAP_AK_not_configured",
            }
            self._record_action("tool_router", "baidu_map_skipped", self.state.map_context)
            return
        try:
            route_top_k = max(1, int(os.getenv("FOODMATE_MAP_ROUTE_TOP_K", "10")))
            original_candidate_count = len(self.state.candidates)
            route_candidates = self.state.candidates
            if original_candidate_count > route_top_k:
                if self.pipeline_mode.endswith("business_rerank") or self.pipeline_mode == "hybrid":
                    route_candidates = rerank(self.state.candidates, self.state.preferences, top_k=route_top_k)
                else:
                    route_candidates = basic_top_k(self.state.candidates, top_k=route_top_k)
            candidates, context = self.map_tool.update_candidate_distances(
                str(user_location), route_candidates, transport_mode=transport_mode
            )
            self.state.candidates = candidates
            self.state.map_context = {
                **context,
                "route_top_k": route_top_k,
                "candidate_count_before_route_prerank": original_candidate_count,
                "candidate_count_after_route_prerank": len(route_candidates),
            }
            self._record_action("tool_router", f"baidu_{transport_mode}_route", self.state.map_context)
        except BaiduMapError as exc:
            self.state.map_context = {
                "enabled": False,
                "distance_source": "school_fallback",
                "reason": str(exc),
            }
            self._record_action("tool_router", "baidu_map_fallback", self.state.map_context)

    def _ranking_tool(self) -> None:
        if self.state.candidates.empty:
            self.state.recommendations = pd.DataFrame()
            return
        candidates = self._attach_feedback_features(self.state.candidates)
        if self.pending_critic_feedback:
            candidates, strategy = self._constrain_candidates(candidates, self.pending_critic_feedback)
            self._record_action("ranking_agent", "critic_feedback_applied", strategy)
            self.pending_critic_feedback = {}
        if self.pipeline_mode.endswith("business_rerank") or self.pipeline_mode == "hybrid":
            recommendations = rerank(candidates, self.state.preferences, top_k=5)
            action = "business_rerank"
        else:
            recommendations = basic_top_k(candidates, top_k=5)
            action = "semantic_topk"
        self.state.recommendations = recommendations
        self._record_action("ranking_agent", action, {"num_recommendations": len(recommendations)})

    def _attach_feedback_features(self, candidates: pd.DataFrame) -> pd.DataFrame:
        memory = self.state.memory or {}
        if candidates.empty:
            return candidates
        liked_restaurants = memory.get("liked_restaurants", {})
        disliked_restaurants = memory.get("disliked_restaurants", {})
        liked_cuisines = memory.get("liked_cuisines", {})
        disliked_cuisines = memory.get("disliked_cuisines", {})
        quality_scores = memory.get("restaurant_quality_scores", {})
        price_sensitivity = float(memory.get("price_sensitivity", 0.5) or 0.5)
        distance_sensitivity = float(memory.get("distance_sensitivity", 0.5) or 0.5)

        def boost(row: pd.Series) -> float:
            name = str(row.get("name", ""))
            cuisine = str(row.get("cuisine", ""))
            score = 0.0
            score += min(0.16, 0.08 * float(liked_restaurants.get(name, 0)))
            score -= min(0.30, 0.15 * float(disliked_restaurants.get(name, 0)))
            score += min(0.10, 0.035 * float(liked_cuisines.get(cuisine, 0)))
            score -= min(0.18, 0.08 * float(disliked_cuisines.get(cuisine, 0)))
            if self.state.preferences.get("budget") and price_sensitivity > 0.5:
                try:
                    price = float(row.get("price_per_person", 0))
                    budget = float(self.state.preferences["budget"])
                    if price > budget:
                        score -= min(0.18, (price_sensitivity - 0.5) * 0.20)
                except (TypeError, ValueError):
                    pass
            if self.state.preferences.get("max_distance_km") and distance_sensitivity > 0.5:
                try:
                    distance = float(row.get("effective_distance_km", row.get("distance_km", 0)))
                    max_distance = float(self.state.preferences["max_distance_km"])
                    if distance > max_distance:
                        score -= min(0.18, (distance_sensitivity - 0.5) * 0.20)
                except (TypeError, ValueError):
                    pass
            return round(score, 4)

        result = candidates.copy()
        result["feedback_boost"] = result.apply(boost, axis=1)
        result["restaurant_quality_score"] = result["name"].map(quality_scores).fillna(0.5).astype(float)
        self._record_action(
            "memory",
            "attach_feedback_features",
            {
                "candidates": len(result),
                "nonzero_feedback_boost": int((result["feedback_boost"] != 0).sum()),
                "quality_scored": int(result["name"].isin(quality_scores.keys()).sum()),
            },
        )
        return result

    def _memory_rerank_tool(self) -> None:
        recs = self.state.recommendations
        if recs.empty:
            return
        liked = set(self.state.memory.get("liked", []))
        disliked = set(self.state.memory.get("disliked", []))
        if not liked and not disliked:
            self._record_action("memory", "memory_rerank_skipped", {"reason": "no_feedback"})
            return
        recs = recs.copy()
        recs["memory_score"] = recs["name"].apply(lambda name: 0.04 if name in liked else (-0.08 if name in disliked else 0.0))
        recs["final_score"] = recs["final_score"].astype(float) + recs["memory_score"]
        self.state.recommendations = recs.sort_values("final_score", ascending=False).reset_index(drop=True)
        self._record_action("memory", "apply_feedback_memory", {"liked": list(liked), "disliked": list(disliked)})

    def _critic_validator(self) -> None:
        report = validate_recommendations(self.state.recommendations, self.state.preferences)
        self.state.critic_report = report
        self._record_action("critic", "validate_top5", report)

    def _repair_tool(self) -> None:
        if self.state.candidates.empty:
            return
        candidates, strategy = self._constrain_candidates(self.state.candidates, self.state.critic_report)
        if candidates.empty:
            # Top30 semantic retrieval can miss an item that satisfies all
            # explicit constraints. Fall back to a deterministic full-KB scan.
            candidates, _ = self._constrain_candidates(self.retriever.df, self.state.critic_report)
            strategy = {
                **strategy,
                "fallback": "structured_full_knowledge_base",
                "fallback_candidate_count": len(candidates),
            }
            if candidates.empty:
                self.state.recommendations = pd.DataFrame()
                self.state.critic_report = {
                    "passed": False,
                    "num_issues": 1,
                    "issues": [{
                        "type": "no_valid_candidates",
                        "message": "没有餐厅同时满足当前菜系和预算约束",
                        "penalty": 0.0,
                    }],
                }
                self._record_action("critic", "repair_no_valid_candidates", strategy)
                return
        self.state.recommendations = rerank(candidates, self.state.preferences, top_k=5)
        self.state.critic_report = validate_recommendations(self.state.recommendations, self.state.preferences)
        detail = {**strategy, "result_count": len(self.state.recommendations), "validation": self.state.critic_report}
        self._record_action("critic", "repair_filter_and_refill", detail)

    def _constrain_candidates(self, candidates: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, dict]:
        """Turn critic issues into deterministic filters over the Top30 pool."""
        constrained = candidates.copy()
        issues = report.get("issues", [])
        issue_types = {issue.get("type") for issue in issues}
        failed_names = {issue.get("name") for issue in issues if issue.get("name")}
        before = len(constrained)

        if failed_names and "name" in constrained.columns:
            constrained = constrained[~constrained["name"].isin(failed_names)]
        # Explicit cuisine is a persistent hard constraint. It must survive later
        # repair rounds that may be triggered by budget or spice issues.
        if self.state.preferences.get("cuisine"):
            constrained = constrained[
                constrained.apply(lambda row: cuisine_match(row, self.state.preferences) >= 0.9, axis=1)
            ]
        if "budget_too_high" in issue_types and self.state.preferences.get("budget"):
            constrained = constrained[
                constrained["price_per_person"].astype(float) <= float(self.state.preferences["budget"]) * 1.15
            ]
        if "spicy_mismatch" in issue_types and self.state.preferences.get("wants_spicy"):
            constrained = constrained[
                constrained.apply(lambda row: spicy_match(row, self.state.preferences) >= 0.9, axis=1)
            ]
        if "avoid_spicy_violation" in issue_types and self.state.preferences.get("avoid_spicy"):
            spicy_pattern = "辣|重口味"
            text_columns = ["cuisine", "tags", "menu_highlights", "review_summary", "pros", "cons"]
            candidate_text = constrained[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
            constrained = constrained[~candidate_text.str.contains(spicy_pattern, regex=True)]

        strategy = {
            "trigger_issue_types": sorted(item for item in issue_types if item),
            "excluded_names": sorted(failed_names),
            "strategy": "hard_filter_and_refill_from_top30",
            "candidate_count_before": before,
            "candidate_count_after": len(constrained),
            "allow_less_than_top5": True,
        }
        return constrained.reset_index(drop=True), strategy


def extract_preferences(text: str) -> dict:
    raw = text
    text = normalize_text(text)
    prefs = {}

    user_location = extract_user_location(raw)
    transport_mode = extract_transport_mode(raw)
    if user_location:
        prefs["user_location"] = user_location
        prefs["transport_mode"] = transport_mode or "walking"
    elif transport_mode:
        prefs["transport_mode"] = transport_mode

    budget_range = extract_budget_range(raw)
    if budget_range:
        prefs["min_budget"] = budget_range[0]
        prefs["budget"] = budget_range[1]

    budget = extract_budget(raw)
    if budget and "budget" not in prefs:
        prefs["budget"] = budget

    distance = extract_distance(raw)
    if distance:
        prefs["max_distance_km"] = distance
    elif contains_any(text, ["离地铁近", "地铁近", "地铁口", "地铁站附近", "离地铁近一点"]):
        prefs["max_distance_km"] = 1.0
    elif contains_any(text, ["near", "close", "campus", "附近", "近一点", "学校近", "离学校近", "周边"]):
        prefs["max_distance_km"] = 2.0

    cuisine_map = {
        "亚洲菜": ["亚洲", "亚洲菜"],
        "粤菜": ["粤菜", "粤式", "早茶", "点心", "顺德菜", "茶餐厅"],
        "客家菜": ["客家菜", "客家", "酿豆腐", "盐焗鸡"],
        "潮汕菜": ["潮汕", "潮菜", "牛肉火锅", "粿条"],
        "浙菜": ["浙菜", "江浙菜", "小海鲜"],
        "火锅": ["火锅", "鸡煲", "涮锅"],
        "中餐": ["中餐", "中国菜", "饺子", "湘菜", "家常菜", "牛肉面"],
        "日料": ["日料", "寿司", "拉面", "便当"],
        "韩餐": ["韩餐", "韩国", "泡菜"],
        "泰餐": ["泰餐", "泰式"],
        "泰国菜": ["泰国菜", "泰国", "东南亚"],
        "越南菜": ["越南", "河粉"],
        "西餐": ["西餐", "西式", "意餐", "披萨", "意面", "法餐", "牛排", "西班牙"],
        "轻食": ["健康", "轻食", "健身", "低脂", "沙拉", "波奇饭"],
        "甜品": ["甜点", "甜品", "蛋糕", "面包", "下午茶", "奶茶", "双皮奶", "舒芙蕾", "漏奶华"],
        "咖啡": ["咖啡", "自习", "甜品"],
        "印度菜": ["印度", "咖喱"],
        "地中海菜": ["地中海"],
    }
    for cuisine, words in cuisine_map.items():
        if contains_any(text, words):
            prefs["cuisine"] = cuisine
            break

    if contains_any(text, ["自己吃", "一个人", "一人食", "单人", "独自", "独自用餐"]):
        prefs["scene"] = "独自用餐"
    elif contains_any(text, ["甜点", "甜品", "蛋糕", "面包", "下午茶", "奶茶"]):
        prefs["scene"] = "独自用餐"
    elif contains_any(text, ["工作日中餐", "工作日午餐", "工作日中午", "中午吃饭", "中饭", "吃中饭", "午饭", "吃午饭"]):
        prefs["scene"] = "快速午餐"
    elif contains_any(text, ["date", "romantic", "约会", "纪念日", "环境好"]):
        prefs["scene"] = "约会"
    elif contains_any(text, ["friends", "group", "朋友", "同学", "聚餐", "多人", "大桌", "家人", "家庭", "小聚", "火锅", "烤肉", "烧烤", "双人套餐", "包厢", "正式"]):
        prefs["scene"] = "朋友聚餐"
    elif contains_any(text, ["quick", "fast", "30 minutes", "赶时间", "快速", "午餐", "中午", "午市", "出餐快", "小吃", "快餐"]):
        prefs["scene"] = "快速午餐"
    elif contains_any(text, ["study", "work", "quiet", "学习", "自习", "办公", "安静", "坐一会儿", "坐坐"]):
        prefs["scene"] = "自习办公"
    elif contains_any(text, ["夜宵", "晚上", "晚餐", "晚上营业", "营业到比较晚", "凌晨"]):
        prefs["scene"] = "朋友聚餐"
    elif contains_any(text, ["healthy", "fitness", "健康", "健身", "低脂", "低卡"]):
        prefs["scene"] = "健康餐"

    if contains_any(text, ["要辣", "辣的", "想吃辣", "能吃辣", "重口味", "麻辣", "香辣", "川菜", "湘菜", "重庆火锅"]):
        prefs["wants_spicy"] = True
        prefs["avoid_spicy"] = False
    if contains_any(text, ["not spicy", "mild", "不要辣", "不辣", "清淡", "别太辣", "不要太辣", "太辣", "少辣"]):
        prefs["avoid_spicy"] = True
        prefs["wants_spicy"] = False
    if contains_any(text, ["vegetarian", "vegan", "素食", "不吃肉", "全素"]):
        prefs["vegetarian"] = True
    if contains_any(
        text,
        [
            "工作日",
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "礼拜一",
            "礼拜二",
            "礼拜三",
            "礼拜四",
            "礼拜五",
            "午市",
            "午餐套餐",
            "双人套餐",
            "单人餐",
            "下午茶",
            "团购",
            "优惠",
            "券",
            "学生",
            "套餐",
        ],
    ):
        prefs["deal_preference"] = True
    if "budget" not in prefs and contains_any(text, ["cheaper", "cheap", "便宜", "更便宜", "平价"]):
        prefs["budget"] = min(prefs.get("budget", 50), 50)
        prefs["deal_preference"] = True
    if "budget" not in prefs and contains_any(text, ["expensive", "高一点", "预算高", "环境好", "正式一点"]):
        prefs["budget"] = max(prefs.get("budget", 150), 150)

    return prefs


def merge_preferences(current: dict, extracted: dict) -> None:
    for key, value in extracted.items():
        if value in [None, "", []]:
            continue
        current[key] = value
    if extracted.get("wants_spicy"):
        current.pop("avoid_spicy", None)
    if extracted.get("avoid_spicy"):
        current.pop("wants_spicy", None)
    if "budget" in extracted and "min_budget" not in extracted:
        current.pop("min_budget", None)


def extract_budget_range(text: str) -> tuple[float, float] | None:
    patterns = [
        r"(?:人均|预算|价格|价位|希望|想要|大概|在)?\s*(\d+)\s*(?:-|~|–|—|到|至|和)\s*(\d+)\s*(?:元|块|人民币)?",
        r"(?:预算|价格|价位).*?(\d+)\s*(?:-|~|–|—|到|至)\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            if low > high:
                low, high = high, low
            return low, high
    return None


def extract_budget(text: str) -> float | None:
    patterns = [
        r"^\s*(\d+(?:\.\d+)?)\s*$",
        r"(?:under|within|below|less than)\s*\$?\s*(\d+)",
        r"\$?\s*(\d+)\s*(?:dollars|usd)",
        r"人均\s*(\d+)",
        r"预算\s*(\d+)",
        r"(\d+)\s*(?:元|块|人民币|美元|刀)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_distance(text: str) -> float | None:
    patterns = [
        r"within\s*(\d+(?:\.\d+)?)\s*km",
        r"(\d+(?:\.\d+)?)\s*km",
        r"(\d+(?:\.\d+)?)\s*公里",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_user_location(text: str) -> str | None:
    patterns = [
        r"(?:我现在在|我在|当前位置(?:是|为)?|位置(?:是|为)?|从)\s*[:：]?\s*([^，。；;]+?)(?=出发|，|。|；|;|$)",
        r"(?:起点(?:是|为)?)\s*[:：]?\s*([^，。；;]+?)(?=，|。|；|;|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip(" ,，。")
            if location and not contains_any(location, ["附近", "周边", "这里", "当前位置"]):
                return location
    return None


def extract_transport_mode(text: str) -> str | None:
    normalized = normalize_text(text)
    if contains_any(normalized, ["骑行", "骑车", "自行车", "单车", "riding", "bike", "bicycle"]):
        return "riding"
    if contains_any(normalized, ["驾车", "开车", "打车", "乘车", "汽车", "网约车", "driving", "drive", "car"]):
        return "driving"
    if contains_any(normalized, ["步行", "走路", "走过去", "walking", "walk"]):
        return "walking"
    return None


def missing_required_fields(preferences: dict) -> list[str]:
    missing = []
    if "budget" not in preferences:
        missing.append("budget")
    if "scene" not in preferences:
        missing.append("scene")
    return missing


def ask_followup(field: str) -> str:
    questions = {
        "budget": "你的大概人均预算是多少？比如 40、80、150 元。",
        "scene": "这次用餐场景是什么？例如朋友聚餐、约会、快速午餐、自习办公或健康轻食。",
    }
    return questions[field]


def render_recommendations(df: pd.DataFrame, preferences: dict) -> str:
    lines = ["我根据你的偏好找到了这些餐厅：", ""]
    for idx, row in df.iterrows():
        distance = float(row.get("effective_distance_km", row["distance_km"]))
        mode = row.get("transport_mode", preferences.get("transport_mode", "walking"))
        mode_labels = {"walking": "步行", "riding": "骑行", "driving": "驾车"}
        is_baidu_route = str(row.get("distance_source", "")).startswith("baidu_")
        distance_label = f"{mode_labels.get(mode, '出行')}距离" if is_baidu_route else "距学校"
        lines.append(
            f"{idx + 1}. {row['name']} - {row['cuisine']}，人均 {row['price_per_person']} 元，"
            f"{distance_label} {distance:.2f} 公里，评分 {row['rating']}。"
        )
        if is_baidu_route and pd.notna(row.get("estimated_duration_min")):
            lines.append(
                f"   预计{mode_labels.get(mode, '出行')}：{float(row['estimated_duration_min']):.0f} 分钟"
            )
        lines.append(f"   推荐理由：{reason_for(row, preferences)}")
        lines.append(f"   可能缺点：{row['cons']}")
        lines.append(f"   引用依据：{row['review_summary']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def reason_for(row: pd.Series, preferences: dict) -> str:
    reasons = []
    if row.get("budget_score", 0) >= 0.9:
        reasons.append("价格符合预算")
    if row.get("distance_score", 0) >= 0.9:
        reasons.append("距离较近")
    if row.get("cuisine_score", 0) >= 0.9:
        reasons.append("菜系匹配")
    if row.get("scene_score", 0) >= 0.9:
        reasons.append("适合当前用餐场景")
    if row.get("rating_score", 0) >= 0.75:
        reasons.append("评分较高")
    if not reasons:
        reasons.append("整体语义匹配度较高")
    reason = "、".join(reasons)
    return reason[:60]


def validate_recommendations(recommendations: pd.DataFrame, preferences: dict) -> dict:
    if recommendations.empty:
        return {"passed": False, "issues": [{"type": "empty_result", "message": "没有可推荐餐厅", "penalty": 0.0}]}

    issues = []
    budget = preferences.get("budget")
    min_budget = preferences.get("min_budget")
    cuisine = preferences.get("cuisine")
    wants_spicy = preferences.get("wants_spicy")
    avoid_spicy = preferences.get("avoid_spicy")

    for _, row in recommendations.iterrows():
        name = row.get("name", "")
        price = float(row.get("price_per_person", 0))
        if budget and price > float(budget) * 1.15:
            issues.append(
                {
                    "type": "budget_too_high",
                    "name": name,
                    "message": f"{name} 人均 {price} 超出预算上限",
                    "penalty": 0.2,
                }
            )
        if min_budget and budget and price < float(min_budget) * 0.75:
            issues.append(
                {
                    "type": "budget_too_low",
                    "name": name,
                    "message": f"{name} 明显低于用户期望价位",
                    "penalty": 0.12,
                }
            )
        if cuisine and float(row.get("cuisine_score", 0.0)) < 0.9:
            issues.append(
                {
                    "type": "cuisine_mismatch",
                    "name": name,
                    "message": f"{name} 与 {cuisine} 偏好不匹配",
                    "penalty": 0.18,
                }
            )
        if wants_spicy and float(row.get("spicy_score", 0.0)) < 0.9:
            issues.append(
                {
                    "type": "spicy_mismatch",
                    "name": name,
                    "message": f"{name} 不符合要辣偏好",
                    "penalty": 0.18,
                }
            )
        text = " ".join(str(row.get(col, "")) for col in ["cuisine", "tags", "menu_highlights", "review_summary", "pros", "cons"])
        if avoid_spicy and ("辣" in text or "重口味" in text) and "不辣" not in text and "清淡" not in text:
            issues.append(
                {
                    "type": "avoid_spicy_violation",
                    "name": name,
                    "message": f"{name} 可能偏辣，不符合忌辣偏好",
                    "penalty": 0.2,
                }
            )

    severe_issues = [issue for issue in issues if issue["type"] in {"budget_too_high", "cuisine_mismatch", "spicy_mismatch", "avoid_spicy_violation"}]
    return {
        "passed": len(severe_issues) == 0,
        "num_issues": len(issues),
        "issues": issues[:10],
    }


def basic_top_k(candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Return semantic Top-K while keeping the columns expected by demos/eval."""
    if candidates.empty:
        return candidates.copy()
    result = candidates.copy()
    if "semantic_similarity" not in result.columns:
        result["semantic_similarity"] = 0.0
    result = result.sort_values("semantic_similarity", ascending=False).head(top_k).reset_index(drop=True)
    result["semantic_score"] = result["semantic_similarity"].astype(float)
    result["budget_score"] = 0.5
    result["distance_score"] = 0.5
    result["travel_time_score"] = 0.5
    result["cuisine_score"] = 0.5
    result["scene_score"] = 0.5
    result["deal_score"] = 0.5
    result["rating_score"] = 0.0
    result["spicy_score"] = 0.5
    result["memory_score"] = 0.0
    result["final_score"] = result["semantic_score"]
    return result

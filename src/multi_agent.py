from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent import FoodMateAgent
from src.rental_agent import RentalAgent
from src.retriever import RestaurantRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "restaurants_cuhksz.csv"
DEFAULT_EVAL = PROJECT_ROOT / "data" / "eval_cases_cuhksz.csv"
SUPPORTED_PIPELINES = [
    "hybrid",
    "bge",
    "bge+cross_encoder",
    "bge+cross_encoder+business_rerank",
]


@dataclass
class MultiAgentState:
    user_query: str = ""
    intent: str = "recommendation"
    active_agent: str = "supervisor"
    plan: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    recommendations: pd.DataFrame = field(default_factory=pd.DataFrame)
    experiment_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    critic_report: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    result: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class SupervisorAgent:
    """Routes requests to a specialized agent and controls retries."""

    def __init__(self, recommendation_agent=None, rental_agent=None, experiment_agent=None, critic_agent=None):
        self.recommendation_agent = recommendation_agent or RecommendationAgent()
        self.rental_agent = rental_agent or RentalRecommendationAgent()
        self.experiment_agent = experiment_agent or ExperimentAgent()
        self.critic_agent = critic_agent or CriticAgent()
        self.state = MultiAgentState()
        self.last_intent: str | None = None
        self.pending_followup_intent: str | None = None
        self.max_retries = int(os.getenv("FOODMATE_AGENT_MAX_RETRIES", "1"))

    def reset(self) -> None:
        self.state = MultiAgentState()
        self.last_intent = None
        self.pending_followup_intent = None
        self.recommendation_agent.reset()
        self.rental_agent.reset()

    def run(self, user_query: str, forced_intent: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        self.state = MultiAgentState(user_query=user_query)
        self.state.intent = forced_intent or self.route(user_query)
        self.last_intent = self.state.intent
        active_agents = {
            "recommendation": "recommendation_agent",
            "rental": "rental_agent",
            "experiment": "experiment_agent",
        }
        self.state.active_agent = active_agents[self.state.intent]
        self.state.plan = self.build_plan(self.state.intent, user_query)
        self._record("supervisor", "route", {"intent": self.state.intent, "active_agent": self.state.active_agent})
        self._record("supervisor", "plan", {"steps": self.state.plan})

        while True:
            if self.state.intent == "experiment":
                agent_result = self.experiment_agent.run(user_query, self.state)
            elif self.state.intent == "rental":
                agent_result = self.rental_agent.run(user_query, self.state)
            else:
                agent_result = self.recommendation_agent.run(user_query, self.state)

            self.state.result = agent_result
            self.state.preferences = agent_result.get("preferences", {})
            self.state.recommendations = agent_result.get("recommendations", pd.DataFrame())
            self.state.metrics = agent_result.get("metrics", {})
            self.state.critic_report = self.critic_agent.validate(self.state.intent, agent_result)
            self._record("critic_agent", "validate", self.state.critic_report)

            if self.state.critic_report.get("passed", False) or self.state.retry_count >= self.max_retries:
                break
            self.state.retry_count += 1
            self._record("supervisor", "retry", {"retry_count": self.state.retry_count})
            if self.state.intent == "recommendation":
                self.recommendation_agent.apply_critic_feedback(self.state.critic_report)
            elif self.state.intent == "rental":
                self.rental_agent.apply_critic_feedback(self.state.critic_report)
            else:
                self.experiment_agent.apply_critic_feedback(self.state.critic_report)

        self.pending_followup_intent = self.state.intent if agent_result.get("type") == "followup" else None
        self.state.latency_ms = (time.perf_counter() - started) * 1000
        return self._build_response()

    def route(self, user_query: str) -> str:
        text = str(user_query).lower()
        experiment_keywords = [
            "ab实验", "a/b", "评估", "跑实验", "对比", "比较pipeline", "比较链路",
            "ndcg", "mrr", "hitrate", "precision", "recall", "latency", "诊断",
            "cross encoder", "crossencoder", "离线实验",
        ]
        if any(keyword in text for keyword in experiment_keywords):
            return "experiment"
        rental_keywords = [
            "租房", "房源", "月租", "租金", "整租", "合租", "公寓", "押一付一",
            "几室", "一室", "二室", "两室", "三室", "四室", "入住", "看房",
            "房子", "小区", "户型", "平方米", "平米",
        ]
        if any(keyword in text for keyword in rental_keywords):
            return "rental"
        recommendation_keywords = [
            "餐厅", "吃饭", "想吃", "推荐菜", "火锅", "咖啡", "甜品", "晚餐",
            "午餐", "中餐", "西餐", "日料", "粤菜", "客家菜", "湘菜", "烤肉",
        ]
        if any(keyword in text for keyword in recommendation_keywords):
            return "recommendation"
        if self.pending_followup_intent in {"rental", "recommendation"}:
            return self.pending_followup_intent
        if self.last_intent == "rental" and re.search(r"\d{3,5}\s*(?:元)?", text):
            return "rental"
        return "recommendation"

    def build_plan(self, intent: str, user_query: str) -> list[str]:
        if intent == "experiment":
            return ["parse_experiment_config", "run_evaluation_tools", "compare_metrics", "critic", "experiment_report"]
        if intent == "rental":
            return ["rental_preference_understanding", "rental_hybrid_retrieval", "rental_business_rerank", "critic", "answer"]
        return ["preference_understanding", "retrieval_tools", "ranking", "memory", "critic", "answer"]

    def _record(self, agent: str, action: str, detail: dict[str, Any] | None = None) -> None:
        self.state.actions.append({"agent": agent, "action": action, "detail": detail or {}})

    def _build_response(self) -> dict[str, Any]:
        result = dict(self.state.result)
        child_actions = result.get("actions", [])
        result.update(
            {
                "supervisor_intent": self.state.intent,
                "active_agent": self.state.active_agent,
                "supervisor_plan": list(self.state.plan),
                "multi_agent_actions": [*self.state.actions, *child_actions],
                "multi_agent_critic": dict(self.state.critic_report),
                "retry_count": self.state.retry_count,
                "latency_ms": self.state.latency_ms,
            }
        )
        return result


class RecommendationAgent:
    """Specialized user-facing recommendation agent."""

    def __init__(self, pipeline_mode: str | None = None):
        self.pipeline_mode = pipeline_mode or os.getenv("FOODMATE_PIPELINE_MODE", "hybrid")
        self.agent = FoodMateAgent(pipeline_mode=self.pipeline_mode)

    def reset(self) -> None:
        self.agent.reset()

    def run(self, user_query: str, state: MultiAgentState) -> dict[str, Any]:
        result = self.agent.handle(user_query)
        state.actions.append(
            {
                "agent": "recommendation_agent",
                "action": "run_stateful_rag",
                "detail": {
                    "pipeline_mode": self.pipeline_mode,
                    "result_type": result.get("type"),
                    "attempt": state.retry_count + 1,
                    "trigger": "critic_retry" if state.retry_count else "initial_request",
                },
            }
        )
        return result

    def apply_critic_feedback(self, report: dict[str, Any]) -> None:
        self.agent.apply_critic_feedback(report)


class RentalRecommendationAgent:
    """Specialized stateful rental recommendation agent."""

    def __init__(self):
        self.agent = RentalAgent()

    def reset(self) -> None:
        self.agent.reset()

    def run(self, user_query: str, state: MultiAgentState) -> dict[str, Any]:
        result = self.agent.handle(user_query)
        state.actions.append(
            {
                "agent": "rental_agent",
                "action": "run_rental_rag",
                "detail": {"result_type": result.get("type")},
            }
        )
        return result

    def apply_critic_feedback(self, report: dict[str, Any]) -> None:
        self.agent.state.critic_report = report


class ExperimentAgent:
    """Runs offline evaluations and diagnoses pipeline trade-offs."""

    def __init__(self):
        self.force_all_pipelines = False

    def run(self, user_query: str, state: MultiAgentState) -> dict[str, Any]:
        config = self.parse_config(user_query)
        if self.force_all_pipelines:
            config["pipelines"] = list(SUPPORTED_PIPELINES)
            self.force_all_pipelines = False
        state.experiment_config = config
        state.actions.append({"agent": "experiment_agent", "action": "parse_config", "detail": config})

        summaries = []
        detailed = {}
        for pipeline in config["pipelines"]:
            result = self.run_evaluation_tool(config, pipeline)
            detailed[pipeline] = result
            summaries.append(self.summarize(pipeline, result))
            state.actions.append(
                {
                    "agent": "experiment_agent",
                    "action": "run_evaluation",
                    "detail": {"pipeline": pipeline, "cases": len(result)},
                }
            )

        table = pd.DataFrame(summaries)
        diagnosis = self.diagnose(table)
        state.metrics = {"table": table.to_dict(orient="records"), "diagnosis": diagnosis}
        state.actions.append({"agent": "experiment_agent", "action": "diagnose_metrics", "detail": diagnosis})
        return {
            "type": "experiment",
            "message": self.render_report(table, diagnosis),
            "preferences": {},
            "recommendations": pd.DataFrame(),
            "metrics": state.metrics,
            "experiment_config": config,
            "actions": [],
        }

    def parse_config(self, user_query: str) -> dict[str, Any]:
        text = str(user_query).lower()
        selected = [pipeline for pipeline in SUPPORTED_PIPELINES if pipeline in text]
        if "cross encoder" in text or "crossencoder" in text:
            selected.append("bge+cross_encoder")
        if "business rerank" in text or "business_rerank" in text or "业务重排" in text:
            selected.append("bge+cross_encoder+business_rerank")
        selected = list(dict.fromkeys(selected))
        if "全部" in text or "四种" in text or "ab" in text or "a/b" in text or not selected:
            selected = list(SUPPORTED_PIPELINES)
        case_limit = None
        match = re.search(r"(?:前|只跑|运行)\s*(\d+)\s*条", text)
        if match:
            case_limit = int(match.group(1))
        return {
            "data_path": str(Path(os.getenv("FOODMATE_DATA_PATH", str(DEFAULT_DATA)))),
            "eval_path": str(DEFAULT_EVAL),
            "pipelines": selected,
            "case_limit": case_limit,
            "comparison_requested": any(
                token in text for token in ["ab", "a/b", "对比", "比较", "哪个效果更好", "孰优"]
            ),
        }

    @staticmethod
    def run_evaluation_tool(config: dict[str, Any], pipeline: str) -> pd.DataFrame:
        from evaluate import evaluate

        result = evaluate(
            eval_path=config["eval_path"],
            data_path=config["data_path"],
            retriever_mode=pipeline,
            case_limit=config.get("case_limit"),
        )
        return result

    @staticmethod
    def summarize(pipeline: str, result: pd.DataFrame) -> dict[str, Any]:
        metric_cols = [
            "hit_rate@5", "precision@5", "recall@5", "mrr@5", "ndcg@5",
            "budget_sat", "distance_sat", "cuisine_sat", "latency_ms",
        ]
        summary = result[metric_cols].mean(numeric_only=True).to_dict()
        summary["pipeline_mode"] = pipeline
        summary["case_count"] = len(result)
        return summary

    @staticmethod
    def diagnose(table: pd.DataFrame) -> dict[str, Any]:
        if table.empty:
            return {"best_quality": None, "fastest": None, "recommendation": "没有实验结果"}
        best = table.sort_values(["ndcg@5", "mrr@5"], ascending=False).iloc[0]
        fastest = table.sort_values("latency_ms").iloc[0]
        recommendation = (
            f"质量优先选择 {best['pipeline_mode']}；"
            f"延迟优先选择 {fastest['pipeline_mode']}。"
        )
        return {
            "best_quality": best["pipeline_mode"],
            "best_ndcg@5": round(float(best["ndcg@5"]), 4),
            "fastest": fastest["pipeline_mode"],
            "fastest_latency_ms": round(float(fastest["latency_ms"]), 2),
            "recommendation": recommendation,
        }

    @staticmethod
    def render_report(table: pd.DataFrame, diagnosis: dict[str, Any]) -> str:
        columns = ["pipeline_mode", "hit_rate@5", "precision@5", "mrr@5", "ndcg@5", "latency_ms"]
        body = table[columns].to_string(index=False) if not table.empty else "没有实验结果"
        return f"实验完成。\n\n{body}\n\n诊断：{diagnosis.get('recommendation', '')}"

    def apply_critic_feedback(self, report: dict[str, Any]) -> None:
        self.force_all_pipelines = True


class CriticAgent:
    """Validates recommendation constraints and experiment comparability."""

    def validate(self, intent: str, result: dict[str, Any]) -> dict[str, Any]:
        if intent == "experiment":
            return self.validate_experiment(result)
        if intent == "rental":
            return self.validate_rental(result)
        return self.validate_recommendation(result)

    @staticmethod
    def validate_rental(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("type") == "followup":
            return {"passed": True, "mode": "rental", "issues": [], "reason": "等待用户补充租房条件"}
        recs = result.get("recommendations", pd.DataFrame())
        required = {"listing_id", "title", "monthly_rent", "final_score", "budget_score", "verified_score"}
        missing = sorted(required.difference(recs.columns)) if not recs.empty else sorted(required)
        issues = []
        if recs.empty:
            issues.append({"type": "empty_recommendations", "message": "租房推荐结果为空"})
        if missing:
            issues.append({"type": "missing_columns", "message": f"缺少字段: {missing}"})
        return {"passed": not issues, "mode": "rental", "issues": issues, "top_k": len(recs)}

    @staticmethod
    def validate_recommendation(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("type") == "followup":
            return {"passed": True, "mode": "recommendation", "issues": [], "reason": "等待用户补充信息"}
        recs = result.get("recommendations", pd.DataFrame())
        child_report = result.get("critic_report", {})
        issues = list(child_report.get("issues", []))
        if recs.empty:
            issues.append({"type": "empty_recommendations", "message": "推荐结果为空"})
        required_cols = {"name", "final_score", "semantic_score", "budget_score", "distance_score"}
        missing_cols = sorted(required_cols.difference(recs.columns)) if not recs.empty else sorted(required_cols)
        if missing_cols:
            issues.append({"type": "missing_columns", "message": f"缺少字段: {missing_cols}"})
        return {
            "passed": len(issues) == 0,
            "mode": "recommendation",
            "issues": issues[:10],
            "top_k": len(recs),
        }

    @staticmethod
    def validate_experiment(result: dict[str, Any]) -> dict[str, Any]:
        metrics = result.get("metrics", {})
        rows = metrics.get("table", [])
        issues = []
        if not rows:
            issues.append({"type": "empty_metrics", "message": "没有生成实验指标"})
        pipelines = [row.get("pipeline_mode") for row in rows]
        comparison_requested = result.get("experiment_config", {}).get("comparison_requested", False)
        if comparison_requested and len(set(pipelines)) < 2:
            issues.append({"type": "not_comparable", "message": "A/B 实验至少需要两个 pipeline"})
        case_counts = {row.get("case_count") for row in rows}
        if len(case_counts) > 1:
            issues.append({"type": "different_case_count", "message": "不同 pipeline 的测试样本数不一致"})
        for row in rows:
            for metric in ["hit_rate@5", "precision@5", "mrr@5", "ndcg@5", "latency_ms"]:
                if row.get(metric) is None:
                    issues.append({"type": "missing_metric", "message": f"{row.get('pipeline_mode')} 缺少 {metric}"})
        return {
            "passed": len(issues) == 0,
            "mode": "experiment",
            "issues": issues,
            "pipeline_count": len(rows),
        }

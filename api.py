from __future__ import annotations

from typing import Optional

from src.feedback import log_feedback, load_feedback
from src.multi_agent import RecommendationAgent, SupervisorAgent
from src.rental_agent import RentalAgent
from src.retriever import RestaurantRetriever

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "FastAPI is not installed. Run: pip install fastapi uvicorn"
    ) from exc


app = FastAPI(title="FoodMate RAG Recommender API", version="0.1.0")


class RecommendRequest(BaseModel):
    query: str
    user_location: Optional[str] = None
    transport_mode: str = "walking"
    top_k: int = 5
    retriever_mode: str = "hybrid"
    pipeline_mode: str = "hybrid"


class FeedbackRequest(BaseModel):
    query: str
    restaurant_name: str
    feedback: str
    reason: Optional[str] = ""


class AgentRequest(BaseModel):
    query: str
    pipeline_mode: str = "hybrid"
    user_location: Optional[str] = None
    transport_mode: str = "walking"


class RentalRecommendRequest(BaseModel):
    query: str
    top_k: int = 5
    commute_destination: Optional[str] = None
    transport_mode: str = "walking"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    base_retriever_mode = "bge" if req.pipeline_mode.startswith("bge") else req.retriever_mode
    retriever = RestaurantRetriever(mode=base_retriever_mode)
    recommendation_agent = RecommendationAgent(pipeline_mode=req.pipeline_mode)
    recommendation_agent.agent.retriever = retriever
    agent = SupervisorAgent(recommendation_agent=recommendation_agent)
    query = req.query
    if req.user_location:
        query = f"{query}。当前位置：{req.user_location}。出行方式：{req.transport_mode}"
    result = agent.run(query, forced_intent="recommendation")
    recs = result["recommendations"].head(req.top_k)
    return {
        "type": result["type"],
        "message": result["message"],
        "preferences": result["preferences"],
        "intent": result.get("intent"),
        "plan": result.get("plan", []),
        "actions": result.get("actions", []),
        "memory": result.get("memory", {}),
        "critic_report": result.get("critic_report", {}),
        "map_context": result.get("map_context", {}),
        "supervisor_intent": result.get("supervisor_intent"),
        "active_agent": result.get("active_agent"),
        "supervisor_plan": result.get("supervisor_plan", []),
        "multi_agent_actions": result.get("multi_agent_actions", []),
        "multi_agent_critic": result.get("multi_agent_critic", {}),
        "retry_count": result.get("retry_count", 0),
        "latency_ms": result.get("latency_ms", 0.0),
        "recommendations": recs.to_dict(orient="records"),
    }


@app.post("/agent")
def run_agent(req: AgentRequest):
    supervisor = SupervisorAgent(
        recommendation_agent=RecommendationAgent(pipeline_mode=req.pipeline_mode)
    )
    query = req.query
    if req.user_location:
        query = f"{query}。当前位置：{req.user_location}。出行方式：{req.transport_mode}"
    result = supervisor.run(query)
    response = dict(result)
    recommendations = response.get("recommendations")
    if recommendations is not None:
        response["recommendations"] = recommendations.to_dict(orient="records")
    return response


@app.post("/rental/recommend")
def rental_recommend(req: RentalRecommendRequest):
    rental_agent = RentalAgent()
    query = req.query
    if req.commute_destination:
        query += f"。通勤目的地：{req.commute_destination}。出行方式：{req.transport_mode}"
    result = rental_agent.handle(query)
    recs = result["recommendations"].head(req.top_k)
    return {
        "type": result["type"],
        "domain": "rental",
        "message": result["message"],
        "preferences": result["preferences"],
        "plan": result.get("plan", []),
        "actions": result.get("actions", []),
        "critic_report": result.get("critic_report", {}),
        "map_context": result.get("map_context", {}),
        "recommendations": recs.to_dict(orient="records"),
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    row = log_feedback(
        query=req.query,
        restaurant_name=req.restaurant_name,
        feedback=req.feedback,
        reason=req.reason or "",
    )
    return {"status": "ok", "feedback": row}


@app.get("/feedback")
def feedback_rows():
    return {"rows": load_feedback().to_dict(orient="records")}

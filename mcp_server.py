from __future__ import annotations

import json
import sys
from typing import Any

from evaluate import evaluate
from src.agent import FoodMateAgent, extract_preferences
from src.retriever import RestaurantRetriever
from src.reranker import rerank


PROTOCOL_VERSION = "2024-11-05"


def read_message() -> dict[str, Any] | None:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("utf-8").strip()
        if not line:
            break
        key, value = line.split(":", 1)
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "extract_preferences",
            "description": "从中文餐厅需求中抽取预算、菜系、场景、距离、忌口等偏好。",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "search_restaurants",
            "description": "从中文餐厅知识库中语义召回候选餐厅。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "recommend_restaurants",
            "description": "执行完整推荐链路：偏好抽取、RAG 召回、多目标重排序、可解释推荐。",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "evaluate_recommender",
            "description": "运行离线评估集，返回 Hit Rate@5、Recall@5、NDCG@5、约束满足率等指标。",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def as_content(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=2)}]}


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "extract_preferences":
        return as_content(extract_preferences(args["query"]))

    if name == "search_restaurants":
        retriever = RestaurantRetriever()
        top_k = int(args.get("top_k", 10))
        rows = retriever.search(args["query"], top_k=top_k).to_dict(orient="records")
        return as_content(rows)

    if name == "recommend_restaurants":
        agent = FoodMateAgent()
        result = agent.handle(args["query"])
        payload = {
            "type": result["type"],
            "message": result["message"],
            "preferences": result["preferences"],
            "recommendations": result["recommendations"].to_dict(orient="records"),
        }
        return as_content(payload)

    if name == "evaluate_recommender":
        result = evaluate()
        summary = result.drop(columns=["case_id", "recommended"]).mean(numeric_only=True).to_dict()
        return as_content({"summary": summary, "cases": result.to_dict(orient="records")})

    raise ValueError(f"Unknown tool: {name}")


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": "foodmate-rag-recommender", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": tool_schema()}
        elif method == "tools/call":
            params = request.get("params", {})
            result = call_tool(params["name"], params.get("arguments", {}))
        else:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}}


def main() -> None:
    while True:
        request = read_message()
        if request is None:
            break
        response = handle(request)
        if response is not None:
            write_message(response)


if __name__ == "__main__":
    main()

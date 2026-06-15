# MCP Tool Server

本项目新增了一个轻量 MCP tool server：

```powershell
python mcp_server.py
```

它通过 MCP 常用的 JSON-RPC + `Content-Length` stdio 消息格式暴露推荐系统能力。

## Tools

### extract_preferences

从中文自然语言需求中抽取偏好：

```json
{
  "query": "今晚想和朋友聚餐，人均80元以内，想吃亚洲菜，不要太辣，离学校近一点。"
}
```

返回：

```json
{
  "budget": 80,
  "max_distance_km": 2.0,
  "cuisine": "亚洲菜",
  "scene": "朋友聚餐",
  "avoid_spicy": true
}
```

### search_restaurants

只做 RAG 召回，返回候选餐厅。

### recommend_restaurants

完整推荐链路：

```text
偏好抽取 -> RAG 召回 -> 多目标重排序 -> 可解释推荐
```

### evaluate_recommender

运行离线评估集，返回：

- `hit_rate@5`
- `recall@5`
- `ndcg@5`
- 偏好抽取准确率
- 预算/距离/菜系约束满足率

## 工具服务说明

```text
Exposed retrieval, reranking, preference extraction, and evaluation modules as MCP-style tools through a JSON-RPC stdio server, enabling protocol-based migration from local function calls to modular agent tool invocation.
```

```text
将偏好抽取、语义召回、多目标重排序和离线评估模块封装为 MCP 风格工具，通过 JSON-RPC stdio 服务暴露，支持从本地函数调用迁移到协议化 Agent 工具调用。
```

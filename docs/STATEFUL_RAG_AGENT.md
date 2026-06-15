# Stateful RAG Agent with Planner, Tools, Critic, and Memory

## 1. 升级目标

原始 FoodMate 更接近固定 workflow：

```text
偏好抽取 → Query Rewrite → Retrieval → Rerank → Answer
```

升级后变成状态机式 RAG Agent：

```text
User Input
↓
Intent Router
↓
Preference Extractor
↓
Planner
↓
Memory Lookup
↓
Clarification Node
↓
Tool Router
  - Retrieval Tool
  - CrossEncoder Tool
  - Ranking Tool
↓
Memory Rerank
↓
Critic Validator
  - pass → Answer
  - fail → Repair → Answer
↓
Answer Generator
```

核心变化：

```text
不是所有输入都盲目走同一条路径，而是基于 AgentState 记录 intent、plan、actions、memory、critic_report，并根据状态决定下一步。
```

## 2. AgentState

代码位置：

```text
src/agent.py
```

AgentState 是整个 Agent 的共享状态。

主要字段：

| 字段 | 作用 |
|---|---|
| `preferences` | 结构化用户偏好，如预算、菜系、场景、辣味 |
| `turns` | 当前会话历史 |
| `intent` | 当前输入意图 |
| `plan` | Planner 生成的节点计划 |
| `actions` | 本轮实际执行过的节点和工具 |
| `memory` | 从 feedback.csv 读取的用户反馈记忆 |
| `candidates` | 检索召回候选 |
| `recommendations` | 最终 Top5 推荐 |
| `critic_report` | Critic 校验结果 |
| `final_query` | Query Rewrite 后的检索 query |

设计说明：

```text
AgentState 作为状态容器，每个节点读写同一个 state，从而记录执行路径、条件跳转和工具调用结果。
```

## 3. Intent Router

作用：

```text
判断用户当前输入属于什么意图。
```

当前支持：

```text
new_recommendation
constraint_update
feedback_or_constraint_update
explain_recommendation
```

例如：

```text
预算80以内 → constraint_update
为什么推荐第一家 → explain_recommendation
第二家太贵了 → feedback_or_constraint_update
想找甜点 → new_recommendation
```

设计说明：

```text
Intent Router 让 Agent 能区分新请求、补充条件、反馈和解释请求，为后续动态决策做准备。
```

## 4. Preference Extractor

作用：

```text
把自然语言需求转换成结构化偏好 JSON。
```

例子：

```text
预算60-110，周一和同学吃中饭，要辣的
```

输出：

```json
{
  "min_budget": 60,
  "budget": 110,
  "scene": "快速午餐",
  "wants_spicy": true,
  "deal_preference": true
}
```

偏好抽取支持：

```text
规则抽取
可选 LLM JSON 抽取
```

LLM 不直接推荐餐厅，只补充结构化偏好。

## 5. Planner

作用：

```text
根据当前 state 生成本轮执行计划。
```

如果缺预算或场景：

```text
memory_lookup → clarification
```

如果信息完整：

```text
memory_lookup
→ clarification
→ query_rewrite
→ retrieval
→ cross_encoder_rerank
→ business_rerank
→ memory_rerank
→ critic_validator
→ answer
```

设计说明：

```text
Planner 不直接生成答案，而是决定这一轮需要经过哪些节点和工具。
```

## 6. Memory Lookup

作用：

```text
读取 reports/feedback.csv 中的历史反馈。
```

当前会提取：

```text
liked：用户喜欢过的餐厅
disliked：用户不喜欢过的餐厅
avoid_reasons：负反馈原因
```

后续在 Memory Rerank 里：

```text
喜欢过的餐厅小幅加分
不喜欢过的餐厅降分
```

设计说明：

```text
Memory 让推荐结果可以被用户反馈影响，而不是每次都从零开始。
```

## 7. Clarification Node

作用：

```text
判断是否需要追问。
```

当前必要字段：

```text
budget
scene
```

如果缺失：

```text
返回 followup，不继续检索。
```

例子：

```text
用户：想吃饭
Agent：你的大概人均预算是多少？
```

## 8. Tool Router

作用：

```text
执行 Planner 选择的工具节点。
```

当前工具包括：

```text
search_restaurants
cross_encoder_rerank
business_rerank
semantic_topk
```

### Retrieval Tool

执行：

```text
Query Rewrite → Retriever.search(top_k=30)
```

支持：

```text
hybrid
bge
```

### CrossEncoder Tool

如果 pipeline 包含：

```text
cross_encoder
```

则执行：

```text
query + restaurant_document → bge-reranker-base → relevance score
```

### Ranking Tool

如果 pipeline 是：

```text
hybrid
bge+cross_encoder+business_rerank
```

则执行 Business Rerank。

否则执行 semantic TopK。

## 9. Critic Validator

作用：

```text
检查 Top5 是否满足用户约束。
```

当前检查：

```text
预算是否超出
是否明显低于预算区间
菜系是否匹配
要辣时是否匹配辣味
忌辣时是否推荐重口味
```

输出：

```json
{
  "passed": true,
  "num_issues": 0,
  "issues": []
}
```

如果失败，会进入 Repair：

```text
对不合格餐厅降分 → 重新排序 → 再校验
```

设计说明：

```text
Critic 是自校验节点，负责检查推荐是否违反用户硬约束。它让 Agent 有反思和修正能力，而不是生成后直接输出。
```

## 10. Answer Generator

作用：

```text
生成最终推荐解释。
```

两种模式：

```text
LLM 已启用 → LLM 生成解释
LLM 未启用 → 模板生成解释
```

页面会显示：

```text
解释来源：LLM 生成
```

或：

```text
解释来源：模板生成
```

## 11. 页面展示

Streamlit 页面现在展示：

```text
Intent
Plan
Critic Report
Memory
Actions
```

这些字段用于观察 Agent 的决策过程：

```text
系统不是单纯 workflow，而是可观测的状态机式 Agent。
```

## 12. 和 ReAct / 多 Agent 的关系

当前实现不是无限自由的 ReAct，而是：

```text
可控状态机 + 有限工具调用 + Critic 校验
```

这样做的原因：

```text
推荐系统需要稳定、可评估、可控。
```

它也不是简单线性多 Agent：

```text
Planner → Retriever → Ranking → Answer
```

而是通过 AgentState 和条件节点实现：

```text
缺信息则追问
信息完整才检索
结果不合格则 repair
反馈会影响排序
```

## 13. 架构摘要

```text
系统使用 AgentState 管理偏好、计划、候选、推荐、反馈记忆和校验报告；Planner 决定本轮需要哪些节点；Tool Router 调用检索、CrossEncoder 和业务重排；Critic 检查 Top5 是否违反预算、菜系、辣味等约束；Memory 从 feedback.csv 读取历史反馈并影响排序。
```

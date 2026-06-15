# FoodMate 项目设计

## 项目范围

FoodMate 面向大学及办公区域周边的本地生活需求，提供餐厅和租房两个领域的自然语言推荐。项目重点是可解释检索与排序、稳定的工具调用、明确的状态管理和可复现评估。

## 设计目标

- 将自然语言需求转换为结构化偏好。
- 在小型结构化知识库上实现可替换的检索链路。
- 对预算、距离、户型和场景等业务约束进行显式建模。
- 对缺失信息进行追问，并保留当前任务的短期状态。
- 通过 Critic 检查结果并在必要时修复。
- 提供 Web、API、协议工具和离线评估入口。

## Agent 组成

- `SupervisorAgent`：识别餐厅、租房或实验任务并分派领域 Agent。
- `RecommendationAgent`：执行餐厅偏好抽取、检索、排序和解释。
- `RentalRecommendationAgent`：执行租房筛选、通勤计算和排序。
- `ExperimentAgent`：运行不同检索/排序组合的离线评估。
- `CriticAgent`：验证结果字段和业务约束。

## 检索与排序

```text
结构化偏好
  -> Query Rewrite
  -> Hybrid/BGE Top30
  -> CrossEncoder
  -> Business Prerank Top10
  -> Optional Map Routing
  -> Final Top5
  -> Critic
```

餐厅重排考虑语义、菜系、预算、距离、出行时间、评分、场景、优惠和辣味。租房重排考虑语义、租金、卧室数、面积、设施、地铁、装修、验真状态和通勤时间。

## 数据

- `restaurants_cuhksz_geo.csv`：餐厅结构化知识库与坐标。
- `rental_listings_cuhksz_geo.csv`：租房结构化知识库与坐标。
- `eval_cases_cuhksz.csv`：餐厅离线测试集。
- `multi_agent_eval_cases.csv`：Agent 路由测试集。

原始网页文本、浏览器登录状态和用户反馈不进入仓库。

## 评估

- Routing Accuracy
- HitRate@5
- Precision@5
- MRR@5
- NDCG@5
- Budget/Cuisine/Distance Satisfaction
- Latency
- Critic Pass Rate

## 接口

- Streamlit：交互式演示。
- FastAPI：`/recommend` 服务接口。
- MCP-style server：偏好抽取、检索、推荐和评估工具。
- Harness：记录路由、计划、工具调用、校验和延迟。

## 当前限制

- 数据规模较小，主要验证链路与业务约束。
- 离线测试集为人工构造，不能替代真实在线指标。
- LLM 与百度地图依赖外部服务，存在延迟、费用和限流。
- `feedback.csv` 目前用于轻量记忆，尚未训练 Learning-to-Rank 模型。

## 后续方向

- 扩展匿名交互数据并训练 Learning-to-Rank。
- 将向量索引迁移至持久化 FAISS、Chroma 或向量数据库。
- 增加缓存、异步批处理、监控和在线实验。
- 对检索、路线和 LLM 调用增加超时、重试与降级策略。

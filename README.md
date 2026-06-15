# FoodMate: Stateful Multi-Agent RAG Life Assistant

FoodMate 是一个面向本地生活场景的中文推荐系统，统一支持餐厅推荐与租房筛选。系统将自然语言偏好抽取、混合检索、CrossEncoder 精排、业务重排、地图通勤计算、反馈记忆和 Critic 校验组织成可观测的 Stateful Multi-Agent RAG 流程。

## 功能

- 餐厅推荐：预算、菜系、距离、场景、辣味、优惠和评分等约束。
- 租房推荐：总租金、合租人数、户型、面积、设施和通勤时间等约束。
- 多轮补充：需求不完整时追问，并在后续轮次合并偏好。
- Hybrid Retrieval：TF-IDF/SVD 或 BGE 向量召回 Top30。
- CrossEncoder：对候选文档进行 query-document 相关性精排。
- Business Rerank：融合语义、预算、距离、户型、场景和反馈特征。
- 百度地图：可选地点解析及步行、骑行、驾车路线计算。
- Critic：检查预算、菜系、户型等明确约束并触发修复。
- 服务接口：Streamlit、FastAPI 和 MCP-style JSON-RPC 工具服务。
- 离线评估：HitRate@5、Precision@5、MRR@5、NDCG@5、约束满足率和延迟。

## 系统流程

```text
User Query
  -> Supervisor / Intent Routing
  -> Restaurant Agent or Rental Agent
  -> Preference Extraction and Clarification
  -> Query Rewrite
  -> Hybrid/BGE Retrieval Top30
  -> CrossEncoder Rerank
  -> Business Prerank Top10
  -> Optional Baidu Route Calculation
  -> Final Rerank Top5
  -> Memory Rerank
  -> Critic Validation and Repair
  -> Grounded Answer
```

## 项目结构

```text
app.py                       Streamlit 应用
api.py                       FastAPI 服务
mcp_server.py                MCP-style 工具服务
multi_agent_harness.py       Agent 路由与任务链路评估
evaluate.py                  离线评估
ab_evaluate.py               Pipeline A/B 对比
data/                        餐厅、租房及评估数据
src/agent.py                 餐厅 Stateful RAG Agent
src/rental_agent.py          租房 RAG Agent
src/multi_agent.py           Supervisor、领域 Agent 与 Critic
src/retriever.py             餐厅检索
src/neural_retrieval.py      BGE 与 CrossEncoder
src/reranker.py              餐厅业务重排
src/tools/baidu_map.py       百度地图工具
tests/                       自动化测试
docs/                        架构与操作文档
reports/                     离线评估结果
```

## 快速开始

建议使用独立 Conda 环境：

```powershell
conda create -n foodmate python=3.10 -y
conda activate foodmate
pip install -r requirements.txt
pip install -r requirements-neural.txt
```

启动 Streamlit：

```powershell
python -m streamlit run app.py
```

启动 FastAPI：

```powershell
python -m uvicorn api:app --reload --port 8000
```

运行测试与评估：

```powershell
python -m unittest discover -s tests -v
python evaluate.py
python ab_evaluate.py
python multi_agent_harness.py --full
```

## 可选配置

真实密钥只通过本机环境变量配置，不要写入代码或提交到 Git。完整变量见 `.env.example`。

```powershell
$env:FOODMATE_USE_LLM="1"
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:FOODMATE_LLM_MODEL="deepseek-v4-flash"
$env:BAIDU_MAP_AK="your-baidu-map-ak"
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_RENTAL_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_RECALL_TOP_K="30"
```

## 数据与隐私

- 仓库只保留推荐所需的结构化公开信息，不保存经纪人姓名、手机号、微信、登录 Cookie 或浏览器 Profile。
- 餐厅和房源价格、营业状态、设施及位置可能变化，使用前应向原平台核验。
- 原始网页、用户反馈、日志、模型缓存和本机密钥由 `.gitignore` 排除。
- 数据和代码仅用于学习、研究与演示，请遵守数据来源网站的服务条款和 robots 规则。

## 文档

- [完整使用流程](docs/FULL_USAGE_GUIDE.md)
- [生活助手使用说明](docs/LIFE_ASSISTANT_USAGE.md)
- [Multi-Agent 架构](docs/MULTI_AGENT_ARCHITECTURE.md)
- [Stateful RAG Agent](docs/STATEFUL_RAG_AGENT.md)
- [神经检索链路](docs/NEURAL_RAG_PIPELINE.md)
- [百度地图接入](docs/BAIDU_MAP_USAGE.md)
- [MCP 工具服务](docs/MCP_USAGE.md)
- [租房数据规范](docs/RENTAL_DATA_SCHEMA.md)

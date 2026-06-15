# FoodMate 简易使用文档

## 1. 项目是什么

FoodMate 是一个中文餐厅 RAG 推荐 Agent。

用户可以输入自然语言需求，例如：

```text
预算60-110，周一和同学吃中饭，要辣的
```

系统会自动理解预算、场景、菜系、优惠、辣味等偏好，并推荐港中深附近的 Top5 餐厅。

## 2. 进入项目目录

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
```

## 3. 激活环境

```powershell
conda activate foodmate
```

如果还没有安装依赖：

```powershell
pip install -r requirements.txt
```

如果要使用 BGE + CrossEncoder 高级链路：

```powershell
pip install -r requirements-neural.txt
```

## 4. 设置数据路径

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
```

## 5. 启动网页 Demo

推荐先用快速模式：

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
python -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 6. 使用高级 RAG 链路

如果要展示 BGE 向量召回 + CrossEncoder 精排：

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python -m streamlit run app.py
```

第一次运行会下载模型，速度较慢。之后模型会缓存在本地。

## 7. 是否使用 LLM

默认不调用 LLM。

如果页面侧边栏显示：

```text
LLM: 未启用
```

说明当前推荐解释由模板生成。

开启 LLM：

```powershell
$env:FOODMATE_USE_LLM="1"
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:FOODMATE_LLM_MODEL="deepseek-v4-flash"
python -m streamlit run app.py
```

开启后页面会显示：

```text
LLM: 已启用
```

推荐结果开头也会显示：

```text
解释来源：LLM 生成
```

## 8. 推荐输入示例

```text
想找个安静的地方坐坐，人均50以内
```

```text
预算60-110，周一和同学吃中饭，要辣的
```

```text
预算79-120，星期五晚上，想找个好吃的甜点
```

```text
中餐，价格50-100元，离学校近一点
```

```text
工作日中午和同学吃饭，想找有套餐或优惠的店，人均80元以内
```

## 9. 页面怎么用

侧边栏会显示：

```text
Pipeline: 当前推荐链路
LLM: 是否启用 LLM
当前抽取偏好
```

如果勾选：

```text
每次输入独立推荐
```

系统会把每次新输入当作一个新需求。

如果取消勾选，系统会保留上一轮偏好，适合连续补充条件。

如果结果混乱，可以点击：

```text
重置对话
```

## 10. Top 推荐表格含义

| 列名 | 含义 |
|---|---|
| `name` | 餐厅名称 |
| `cuisine` | 菜系 |
| `price_per_person` | 人均消费 |
| `rating` | 餐厅评分 |
| `distance_km` | 距离 |
| `final_score` | 最终综合分，排序主要依据 |
| `semantic_score` | 用户需求与餐厅文本的语义相关分 |
| `budget_score` | 预算匹配分 |
| `distance_score` | 距离匹配分 |
| `cuisine_score` | 菜系匹配分 |
| `scene_score` | 场景匹配分 |
| `deal_score` | 优惠/套餐匹配分 |

## 11. 运行命令行 Demo

```powershell
python run_demo.py
```

## 12. 运行 A/B 评估

```powershell
python ab_evaluate.py --data data\restaurants_cuhksz.csv --eval data\eval_cases_cuhksz.csv
```

输出文件：

```text
reports\ab_evaluation_cuhksz.csv
```

A/B 会比较：

```text
hybrid
bge
bge+cross_encoder
bge+cross_encoder+business_rerank
```

## 13. 启动 API

```powershell
python -m uvicorn api:app --reload --port 8000
```

打开：

```text
http://127.0.0.1:8000/docs
```

主要接口：

```text
POST /recommend
POST /feedback
GET  /feedback
GET  /health
```

## 14. 常见问题

### 为什么第一次运行很慢？

如果使用 BGE 或 CrossEncoder，第一次会下载模型。下载完成后会缓存在本地。

### 为什么推荐结果有时候慢？

`bge+cross_encoder+business_rerank` 会对 Top30 候选做 CrossEncoder 精排，CPU 上会比较慢。日常演示建议使用：

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
```

### 为什么页面一直追问预算或场景？

说明当前输入缺少必要信息。可以直接补充：

```text
50
```

或：

```text
自习办公
```

### 为什么结果和上一次输入有关？

如果取消勾选“每次输入独立推荐”，系统会保留上一轮偏好。想重新开始可以点击“重置对话”。

## 15. 推荐演示顺序

1. 启动 Streamlit 页面。
2. 输入“想找个安静的地方坐坐，人均50以内”。
3. 展示当前抽取偏好和 Top 推荐表格。
4. 输入“预算60-110，周一和同学吃中饭，要辣的”。
5. 展示辣味、预算、工作日偏好如何影响排序。
6. 运行 A/B 评估，展示项目不是只做 demo，还有评估 harness。

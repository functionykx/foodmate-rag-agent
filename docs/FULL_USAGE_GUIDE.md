# FoodMate 中文餐厅推荐 Agent 全流程使用文档

## 1. 项目定位

FoodMate 是一个面向中文本地生活场景的 RAG 餐厅推荐 Agent。

当前版本已经支持：

- 中文自然语言需求理解
- 香港中文大学（深圳）附近真实餐厅数据
- 餐厅语义召回
- 多目标重排序
- 优惠/团购/工作日套餐偏好
- 可解释 Top-5 推荐
- 离线评估指标
- Streamlit 网页 demo
- MCP 风格工具服务

## 2. 项目目录

```text
rag_recommender_agent/
  app.py                         Streamlit 网页 demo
  web_demo.py                    无额外前端依赖的网页 demo
  run_demo.py                    命令行 demo
  evaluate.py                    离线评估脚本
  mcp_server.py                  MCP 风格工具服务
  requirements.txt               Python 依赖
  QUICKSTART.md                  快速启动说明

  data/
    restaurants.csv              原始 mock 餐厅数据
    restaurants_cuhksz.csv       港中深附近真实餐厅数据
    eval_cases.csv               mock 数据评估集
    eval_cases_cuhksz.csv        港中深真实数据评估集
    dianping_restaurants_template.csv  大众点评风格数据模板

  src/
    agent.py                     Agent 主逻辑：偏好抽取、追问、推荐生成
    retriever.py                 RAG 语义召回
    reranker.py                  多目标重排序
    prompts.py                   检索 query 构造
    utils.py                     数据读取、文档构造、路径配置
    ingest.py                    构建检索索引
    data_sources/
      dianping_csv_import.py     大众点评风格 CSV 转标准数据格式

  reports/
    evaluation_results.csv       mock 数据评估结果
    evaluation_results_cuhksz.csv 港中深真实数据评估结果

  docs/
    MCP_USAGE.md                 MCP 使用说明
    FULL_USAGE_GUIDE.md          本文档
```

## 3. 环境准备

进入项目目录：

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
```

安装依赖：

```powershell
pip install -r requirements.txt
```

若 Anaconda 环境已安装 `streamlit`、`pandas` 和 `scikit-learn`，可直接运行。

检查 Streamlit：

```powershell
python -m streamlit --version
```

## 4. 选择数据源

默认数据是 mock 餐厅：

```text
data/restaurants.csv
```

如果要使用港中深附近真实餐厅数据，设置环境变量：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
```

这个环境变量只在当前 PowerShell 窗口有效。关闭窗口后需要重新设置。

检查是否能读取真实数据：

```powershell
python -c "from src.utils import load_restaurants; df=load_restaurants(); print(df[['name','cuisine','price_per_person']].head())"
```

## 5. 构建检索索引

构建当前数据源的 TF-IDF 检索索引：

```powershell
python src\ingest.py
```

输出目录：

```text
vector_store/
```

注意：当前代码运行时也能即时构建检索器；`ingest.py` 主要用于展示标准 RAG 项目的索引构建流程。

## 6. 命令行 Demo

运行：

```powershell
python run_demo.py
```

推荐先使用港中深真实数据：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
python run_demo.py
```

你会看到：

- 用户需求
- Top-5 推荐餐厅
- 推荐理由
- 可能缺点
- 引用依据
- Score table

Score table 字段含义：

```text
name              餐厅名称
final_score       综合排序分
semantic_score    RAG 语义相似度
budget_score      预算匹配分
distance_score    距离匹配分
cuisine_score     菜系匹配分
scene_score       场景匹配分
deal_score        优惠/套餐匹配分
```

## 7. Streamlit 网页 Demo

启动：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
python -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

推荐测试输入：

```text
工作日中午和同学吃饭，想找有套餐或优惠的店，人均80元以内，港中深附近。
```

```text
晚上想和朋友吃火锅，人均110元以内，离地铁近一点。
```

```text
下午想找地方自习办公，预算50元以内，最好有咖啡。
```

```text
想找一家适合约会的西餐，预算120元以内，环境好一点。
```

## 8. 无 Streamlit 网页 Demo

如果 Streamlit 无法启动，可以运行：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
python web_demo.py
```

浏览器打开：

```text
http://127.0.0.1:8000
```

这是备用网页 demo，只依赖 Python 标准库。

## 9. 离线评估

港中深真实数据评估：

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --out reports\evaluation_results_cuhksz.csv
```

mock 数据评估：

```powershell
python evaluate.py `
  --data data\restaurants.csv `
  --eval data\eval_cases.csv `
  --out reports\evaluation_results.csv
```

主要指标：

```text
hit_rate@5
Top 5 是否命中至少一个人工标注的合适餐厅。

recall@5
人工标注的合适餐厅中，有多少被 Top 5 找回。

ndcg@5
推荐排序质量，越接近 1 越好。

budget_extract_acc
预算抽取是否正确。

cuisine_extract_acc
菜系抽取是否正确。

scene_extract_acc
场景抽取是否正确。

budget_sat
Top 5 中满足预算约束的比例。

distance_sat
Top 5 中满足距离约束的比例。

cuisine_sat
Top 5 中满足菜系约束的比例。

mrr@5
第一个命中餐厅的倒数排名，越高说明越早把合适结果排到前面。

precision@5
Top 5 中有多少比例属于人工标注的相关餐厅。

latency_ms
单条请求从 Agent 开始处理到返回结果的耗时。
```

当前港中深真实数据版大致结果：

```text
Hit Rate@5: 1.000
Recall@5: 0.894
MRR@5: 1.000
NDCG@5: 0.906
预算满足率: 0.960
距离满足率: 1.000
平均延迟: 约 14 ms
```

## 9.5 A/B 评估

比较 TF-IDF、Embedding fallback、Hybrid Retrieval：

```powershell
python ab_evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --out reports\ab_evaluation_cuhksz.csv
```

结果文件：

```text
reports\ab_evaluation_cuhksz.csv
```

当前 A/B 结果显示 Hybrid 的 `ndcg@5` 较高，说明融合召回对排序质量有帮助。

## 10. MCP 工具服务

启动 MCP server：

```powershell
python mcp_server.py
```

当前暴露工具：

```text
extract_preferences
search_restaurants
recommend_restaurants
evaluate_recommender
```

说明文档：

```text
docs\MCP_USAGE.md
```

## 10.1 FastAPI 服务

安装依赖：

```powershell
pip install fastapi uvicorn
```

启动：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
python -m uvicorn api:app --reload --port 8001
```

推荐接口：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8001/recommend" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"工作日中午和同学吃饭，想找有套餐或优惠的店，人均80元以内","top_k":5,"retriever_mode":"hybrid"}'
```

反馈接口：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8001/feedback" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"工作日中午和同学吃饭","restaurant_name":"文通冰室(大运天地店)","feedback":"like","reason":"套餐便宜"}'
```

反馈会写入：

```text
reports\feedback.csv
```

## 10.5 可选 LLM 增强

项目支持可选 LLM 层。LLM 不负责直接决定推荐结果，而是用于：

```text
1. 更复杂的中文偏好抽取
2. 更自然的推荐解释生成
```

开启方式：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:FOODMATE_USE_LLM="1"
$env:FOODMATE_LLM_MODEL="gpt-4o-mini"
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
python run_demo.py
```

启动网页：

```powershell
python -m streamlit run app.py
```

关闭 LLM：

```powershell
$env:FOODMATE_USE_LLM="0"
```

详细说明见：

```text
docs\LLM_USAGE.md
```

项目展示时可以这样讲：

```text
系统将偏好抽取、餐厅检索、推荐重排和离线评估封装为 MCP 风格工具，使 Agent 能从本地函数调用迁移到协议化工具调用。
```

## 11. 接入更多大众点评数据

不要写绕过登录、验证码、反爬的代码。建议手动或合规整理成 CSV。

模板：

```text
data\dianping_restaurants_template.csv
```

字段：

```text
店名,菜系,人均,评分,距离公里,位置,营业时间,标签,适合场景,推荐菜,评论摘要,优点,缺点,来源
```

转换为项目标准格式：

```powershell
python src\data_sources\dianping_csv_import.py `
  --input data\dianping_restaurants_template.csv `
  --output data\restaurants_real.csv
```

使用新数据：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_real.csv"
python run_demo.py
```

## 12. 修改推荐逻辑

偏好抽取：

```text
src\agent.py
```

例如增加“夜宵”“情侣约会”“学生优惠”等关键词。

召回逻辑：

```text
src\retriever.py
```

当前使用中文字符 n-gram TF-IDF。

重排序逻辑：

```text
src\reranker.py
```

当前综合分：

```text
final_score =
  semantic_score
  + budget_score
  + distance_score
  + cuisine_score
  + scene_score
  + deal_score
  - penalty
```

如果想让优惠权重更高，可以调整 `deal_score` 的权重。

## 13. 常见问题

### PowerShell 设置环境变量

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
```

### CMD 设置环境变量

本项目文档统一使用 PowerShell。CMD 写法不同，不建议混用。

### Streamlit 找不到

```powershell
python -m streamlit --version
pip install streamlit
```

### 中文显示乱码

优先使用 PowerShell 或 Anaconda Prompt，并确保代码文件按 UTF-8 保存。

### 推荐结果不符合预期

检查：

```text
1. 用户需求是否被正确抽取
2. data/restaurants_cuhksz.csv 中是否有对应标签
3. reranker.py 中对应 score 权重是否合适
4. eval_cases_cuhksz.csv 中人工标注是否合理
```

## 14. 推荐展示流程

1. 打开 Streamlit 页面。
2. 输入“工作日套餐/学生低预算”需求。
3. 展示系统推荐绿茶、文通冰室、Thy yeah 等有优惠的店。
4. 展示推荐理由来自真实餐厅字段。
5. 展示分数表：`budget_score`、`distance_score`、`deal_score`。
6. 运行 `evaluate.py`，展示离线指标。
7. 说明 MCP server 将推荐链路封装为工具。

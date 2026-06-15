# FoodMate Neural RAG Pipeline

## 1. 完整流程

```text
用户输入
↓
规则/LLM 偏好抽取
↓
Query Rewrite
↓
Hybrid Retrieval Top30
  - BM25/TF-IDF
  - bge-base-zh-v1.5 / bge-large-zh-v1.5 / bge-m3 dense vector
↓
CrossEncoder Rerank Top30
  - bge-reranker-base / bge-reranker-large / bge-reranker-v2-m3
↓
Business Rerank
  - 预算
  - 距离
  - 菜系
  - 场景
  - 优惠
  - 是否偏辣
↓
Top5
↓
LLM 生成 60 字以内解释
```

## 2. 每一步做什么

### 2.1 用户输入

用户用自然语言描述需求，例如：

```text
工作日中午和同学吃饭，想找有套餐或优惠的店，人均80元以内，港中深附近。
```

这一层输入往往不规整，里面可能同时包含预算、时间、人数、菜系、距离、优惠、忌口。

### 2.2 规则/LLM 偏好抽取

代码位置：

```text
src/agent.py
src/llm.py
```

规则抽取负责稳定识别常见字段：

```text
budget
max_distance_km
cuisine
scene
avoid_spicy
vegetarian
deal_preference
```

LLM JSON 抽取是可选增强，用来理解更复杂的中文表达。它不直接决定最终餐厅，只补全结构化偏好，降低误解用户需求的概率。

### 2.3 Query Rewrite

代码位置：

```text
src/prompts.py
```

系统会把用户原句和抽取出来的偏好拼成更适合检索的 query。

例如用户说“学生工作日午餐”，Query Rewrite 会补充：

```text
优惠 团购 套餐 工作日 午市 学生
```

这样可以提高带套餐、代金券、下午茶团购等餐厅的召回概率。

### 2.4 Hybrid Retrieval Top30

代码位置：

```text
src/retriever.py
src/neural_retrieval.py
```

召回层的目标是从餐厅或租房知识库里先找出一批可能相关的候选，当前小型数据集取 Top30。

TF-IDF 适合精确词匹配：

```text
工作日套餐
粤式茶点
大运中心站C口
```

BGE dense vector 适合理解语义接近的表达：

```text
适合自习办公 ≈ 安静咖啡店
朋友聚餐 ≈ 多人/大桌/火锅/烤肉
学生省钱 ≈ 团购/套餐/低人均
```

当前代码新增了真实 BGE embedding：

```text
sentence-transformers
BAAI/bge-base-zh-v1.5
FAISS
```

默认模型是 `BAAI/bge-base-zh-v1.5`，可以通过环境变量切换：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-large-zh-v1.5"
```

如果没有安装 `sentence-transformers` 或 `faiss-cpu`，系统会自动退回原来的 SVD dense fallback，不会中断 demo。

### 2.5 CrossEncoder Rerank Top30

代码位置：

```text
src/neural_retrieval.py
src/agent.py
```

Bi-encoder/BGE embedding 适合快速召回，但它是分别编码 query 和 document。CrossEncoder 会把 query 和每个候选餐厅成对输入模型，判断二者是否真的匹配。

推荐链路：

```text
BGE 召回 Top30
↓
CrossEncoder 精排 Top30
↓
取更可靠的语义顺序
```

默认模型是：

```text
BAAI/bge-reranker-base
```

可以切换为：

```powershell
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-large"
```

或者：

```powershell
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
```

CrossEncoder 当前只重排 Top30，不直接扫描全库。数据量扩大后可通过环境变量提高召回数，但全库 CrossEncoder 成本仍然过高。

### 2.6 Business Rerank

代码位置：

```text
src/reranker.py
```

CrossEncoder 只解决“语义相关”，Business Rerank 解决“是否真的适合用户”。

业务排序特征包括：

```text
semantic_score
budget_score
distance_score
cuisine_score
scene_score
deal_score
rating_score
dietary_penalty
```

例如：

```text
用户预算80元以内
```

即使一家店语义很相关，如果人均 130 元，也应该被降权。

再比如：

```text
用户说工作日午市套餐
```

带“工作日套餐、午市代金券、下午茶团购”的店会通过 `deal_score` 获得更高排序。

### 2.7 Top5

最终输出 5 家餐厅，适合 demo 展示，也符合真实推荐产品常见的列表长度。

Top5 需要同时满足：

```text
语义相关
预算合理
距离可接受
场景匹配
解释可信
```

### 2.8 LLM 生成 60 字以内解释

代码位置：

```text
src/llm.py
src/agent.py
```

LLM 只负责把已有证据组织成自然语言，不允许编造餐厅信息。

每家店解释限制在 60 字以内，原因是：

```text
便于页面展示
便于快速阅读
降低幻觉和废话
突出核心推荐理由
```

## 3. 安装真实 BGE/FAISS/CrossEncoder

建议使用独立环境：

```powershell
conda create -n foodmate python=3.10 -y
conda activate foodmate
cd "<PROJECT_ROOT>\rag_recommender_agent"
pip install -r requirements.txt
pip install -r requirements-neural.txt
```

无 GPU 环境建议先使用 base 模型：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
```

如果有 GPU，且需要更强模型：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-large-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-large"
```

## 4. 运行四组 A/B

```powershell
python ab_evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --out reports\ab_evaluation_cuhksz.csv
```

A/B 表现在比较：

```text
hybrid
bge
bge+cross_encoder
bge+cross_encoder+business_rerank
```

## 5. 单独运行某条链路

只跑 BGE：

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge
```

BGE + CrossEncoder：

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge+cross_encoder
```

BGE + CrossEncoder + Business Rerank：

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge+cross_encoder+business_rerank
```

## 6. 分层设计摘要

```text
推荐系统拆分为召回、语义精排、业务重排和解释生成四层。
召回层用 Hybrid Retrieval 保证覆盖率，BGE 负责语义召回，TF-IDF 保留关键词精确匹配。
CrossEncoder 负责提升 Top30 内部相关性。
Business Rerank 把预算、距离、优惠、忌口等真实业务约束纳入排序。
最后 LLM 只生成基于证据的短解释，不直接决定推荐结果。
```

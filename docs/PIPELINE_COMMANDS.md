# FoodMate 不同推荐链路运行指令

本文档用于快速切换和运行 FoodMate 的不同推荐链路。

所有命令均使用 PowerShell。

## 1. 通用准备

先进入项目目录：

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
```

激活环境：

```powershell
conda activate foodmate
```

设置港中深餐厅数据：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
```

如果要使用 BGE / CrossEncoder，确保已经安装：

```powershell
pip install -r requirements-neural.txt
```

## 2. 组合一：hybrid

### 链路内容

```text
规则/LLM 偏好抽取
→ Query Rewrite
→ TF-IDF 检索
→ 本地 SVD dense fallback
→ Hybrid 分数融合
→ Business Rerank
→ Top5
```

### 适合场景

```text
日常网页 demo
快速演示
不想等待 BGE / CrossEncoder
普通中文需求测试
```

### Streamlit 网页运行

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
python -m streamlit run app.py
```

### 命令行 demo

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
python run_demo.py
```

### 单独评估

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever hybrid `
  --out reports\evaluation_hybrid.csv
```

## 3. 组合二：bge

### 链路内容

```text
规则/LLM 偏好抽取
→ Query Rewrite
→ BGE embedding
→ FAISS 向量检索
→ Top5
```

### 适合场景

```text
展示真实中文向量检索
测试语义召回能力
验证 BGE embedding 是否生效
```

### 设置模型

推荐 CPU 使用 base：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
```

如果机器性能较好，也可以使用 large：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-large-zh-v1.5"
```

### Streamlit 网页运行

```powershell
$env:FOODMATE_PIPELINE_MODE="bge"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
python -m streamlit run app.py
```

### 命令行 demo

```powershell
$env:FOODMATE_PIPELINE_MODE="bge"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
python run_demo.py
```

### 单独评估

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge `
  --out reports\evaluation_bge.csv
```

## 4. 组合三：bge+cross_encoder

### 链路内容

```text
规则/LLM 偏好抽取
→ Query Rewrite
→ BGE embedding + FAISS 召回 Top30
→ CrossEncoder Rerank Top30
→ Top5
```

### 适合场景

```text
展示语义精排
比较 BGE 和 CrossEncoder 的效果差异
说明为什么需要 reranker
```

### 设置模型

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
```

如果机器性能较好：

```powershell
$env:FOODMATE_BGE_MODEL="BAAI/bge-large-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-large"
$env:FOODMATE_RECALL_TOP_K="30"
```

### Streamlit 网页运行

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python -m streamlit run app.py
```

### 命令行 demo

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python run_demo.py
```

### 单独评估

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge+cross_encoder `
  --out reports\evaluation_bge_cross_encoder.csv
```

## 5. 组合四：bge+cross_encoder+business_rerank

### 链路内容

```text
规则/LLM 偏好抽取
→ Query Rewrite
→ BGE embedding + FAISS 召回 Top30
→ CrossEncoder Rerank Top30
→ Business Rerank
→ Top5
```

### 适合场景

```text
完整链路验证
项目报告
展示完整推荐系统链路
展示 RAG + 推荐系统 + 业务重排
```

### Streamlit 网页运行

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python -m streamlit run app.py
```

### 命令行 demo

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python run_demo.py
```

### 单独评估

```powershell
python evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --retriever bge+cross_encoder+business_rerank `
  --out reports\evaluation_bge_cross_encoder_business.csv
```

## 6. 四组 A/B 一键评估

运行：

```powershell
python ab_evaluate.py `
  --data data\restaurants_cuhksz.csv `
  --eval data\eval_cases_cuhksz.csv `
  --out reports\ab_evaluation_cuhksz.csv
```

该命令会自动比较：

```text
hybrid
bge
bge+cross_encoder
bge+cross_encoder+business_rerank
```

输出：

```text
reports\ab_evaluation_cuhksz.csv
```

## 7. 开启 LLM

默认不调用 LLM。

如需开启 DeepSeek API：

```powershell
$env:FOODMATE_USE_LLM="1"
$env:OPENAI_API_KEY="你的 DeepSeek API Key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:FOODMATE_LLM_MODEL="deepseek-v4-flash"
```

然后运行任意链路，例如：

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
python -m streamlit run app.py
```

关闭 LLM：

```powershell
Remove-Item Env:\FOODMATE_USE_LLM
```

或重新打开一个新的 PowerShell 窗口。

## 8. 推荐演示顺序

### 快速演示

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
python -m streamlit run app.py
```

输入：

```text
想找个安静的地方坐坐，人均50以内
```

### 高级链路演示

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
$env:FOODMATE_BGE_MODEL="BAAI/bge-base-zh-v1.5"
$env:FOODMATE_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:FOODMATE_RECALL_TOP_K="30"
python -m streamlit run app.py
```

输入：

```text
预算60-110，周一和同学吃中饭，要辣的
```

### LLM 演示

```powershell
$env:FOODMATE_USE_LLM="1"
$env:OPENAI_API_KEY="你的 DeepSeek API Key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:FOODMATE_LLM_MODEL="deepseek-v4-flash"
$env:FOODMATE_PIPELINE_MODE="hybrid"
python -m streamlit run app.py
```

页面侧边栏会显示：

```text
LLM: 已启用
```

## 9. 注意事项

### BGE 第一次运行很慢

第一次运行 BGE 或 CrossEncoder 会下载模型，之后会缓存到本地。

### CrossEncoder 比较慢

`bge+cross_encoder` 和 `bge+cross_encoder+business_rerank` 会对 Top30 候选逐个精排，CPU 上可能需要数秒。

### 日常试用建议

```powershell
$env:FOODMATE_PIPELINE_MODE="hybrid"
```

### 完整链路运行

```powershell
$env:FOODMATE_PIPELINE_MODE="bge+cross_encoder+business_rerank"
```

### 如果结果和上一轮输入有关

点击页面里的：

```text
重置对话
```

或勾选：

```text
每次输入独立推荐
```

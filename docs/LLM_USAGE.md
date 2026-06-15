# FoodMate LLM 接入说明

## 1. 是否需要加入 LLM

建议加入，但不要让 LLM 替代推荐系统核心逻辑。

推荐架构：

```text
LLM 负责：
- 复杂中文需求理解
- 结构化偏好抽取
- 自然语言推荐解释
- 多轮追问措辞

规则/RAG/Reranker 负责：
- 餐厅召回
- 预算、距离、菜系、优惠等硬约束
- 多目标排序
- 离线评估
```

这样项目既有 Agent 能力，也保留推荐系统的可控性和可评估性。

## 2. 当前代码如何接入 LLM

新增文件：

```text
src/llm.py
```

它提供：

```text
extract_preferences_with_llm()
generate_recommendation_text()
```

主链路在：

```text
src/agent.py
```

逻辑是：

```text
规则偏好抽取
  ↓
如果 FOODMATE_USE_LLM=1 且配置 OPENAI_API_KEY
  ↓
调用 LLM 修正/补充偏好
  ↓
RAG 召回
  ↓
Reranker 排序
  ↓
如果启用 LLM，用 LLM 生成自然语言推荐解释
  ↓
否则使用规则模板解释
```

没有 API key 时，系统自动回退到原来的非 LLM 版本。

## 3. 配置 OpenAI API

PowerShell 中设置：

```powershell
$env:OPENAI_API_KEY="你的 API Key"
$env:FOODMATE_USE_LLM="1"
$env:FOODMATE_LLM_MODEL="gpt-4o-mini"
```

使用港中深数据：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz.csv"
```

运行命令行 demo：

```powershell
python run_demo.py
```

启动网页 demo：

```powershell
python -m streamlit run app.py
```

## 4. 使用 OpenAI-compatible 服务

使用其他兼容 OpenAI Chat Completions 格式的服务时，可设置：

```powershell
$env:OPENAI_BASE_URL="https://your-provider.example.com/v1"
$env:OPENAI_API_KEY="你的 API Key"
$env:FOODMATE_LLM_MODEL="你的模型名"
$env:FOODMATE_USE_LLM="1"
```

## 5. 关闭 LLM

```powershell
$env:FOODMATE_USE_LLM="0"
```

或者开启一个新的 PowerShell 窗口，不设置 `FOODMATE_USE_LLM`。

## 6. 为什么不让 LLM 直接推荐

不建议：

```text
用户问题 -> LLM 直接推荐
```

原因：

- 容易编造不存在的餐厅、菜单、优惠。
- 很难保证预算、距离、菜系约束。
- 不方便计算 Hit@5、NDCG@5、约束满足率。
- 推荐逻辑难以复现和审计。

推荐：

```text
用户问题 -> LLM/规则抽取偏好 -> RAG 召回 -> Rerank -> LLM 解释
```

## 7. 架构原则

```text
LLM 位于需求理解和解释生成层。核心推荐由 RAG 召回和多目标重排序控制，以保持可解释、可评估和约束可控。
```

## 8. 注意事项

- 不要把 API key 写进代码。
- 不要把用户隐私信息写进日志。
- 评估时可以分别比较 LLM off / LLM on。
- 如果 LLM 返回失败，系统会自动回退到规则模板。

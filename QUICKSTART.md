# FoodMate Quickstart

## 1. 项目是什么

FoodMate 是一个中文本地生活场景的 RAG 餐厅推荐 Agent。

它模拟大众点评/美团/外卖推荐业务：

- 用户用中文自然语言说需求。
- Agent 抽取预算、菜系、距离、用餐场景、忌口等约束。
- Retriever 从中文餐厅知识库召回候选餐厅。
- Reranker 用多目标分数排序。
- Agent 输出 Top-5 推荐、推荐理由、缺点和引用依据。
- MCP server 将偏好抽取、检索、推荐和评估暴露为协议化工具。

## 2. 文件作用

```text
data/restaurants.csv      中文 mock 餐厅知识库
data/eval_cases.csv       中文离线评估集
src/utils.py              数据读取和文档构造，支持 FOODMATE_DATA_PATH 外部数据路径
src/retriever.py          中文字符 n-gram TF-IDF 语义召回
src/reranker.py           多目标推荐排序
src/agent.py              多轮 Agent 和偏好抽取
src/prompts.py            查询构造和 prompt 风格
src/ingest.py             构建本地检索索引
app.py                    Streamlit demo
web_demo.py               无额外依赖的网页 demo
run_demo.py               命令行 demo
evaluate.py               离线评估脚本
mcp_server.py             MCP 风格 tool server
docs/MCP_USAGE.md         MCP 使用说明
```

## 3. 接入真实餐厅数据

不建议在项目里写绕过大众点评反爬、验证码或登录限制的爬虫。更稳妥的方式是：

1. 使用平台允许的开放接口或授权数据。
2. 手动整理你学校附近餐厅信息。
3. 保存网页后做本地解析。
4. 将合法获取的数据整理成 `data/restaurants.csv` 的字段格式。

字段需要保持：

```text
id,name,cuisine,price_per_person,rating,distance_km,location,opening_hours,tags,best_for,menu_highlights,review_summary,pros,cons,source
```

你也可以不覆盖默认文件，而是设置环境变量：

```powershell
$env:FOODMATE_DATA_PATH="C:\path\to\your_restaurants.csv"
python run_demo.py
```

对于类似大众点评字段的中文 CSV，可先按模板填写：

```text
data/dianping_restaurants_template.csv
```

然后转换成项目标准格式：

```powershell
python src\data_sources\dianping_csv_import.py `
  --input data\dianping_restaurants_template.csv `
  --output data\restaurants_real.csv
```

再用真实数据运行：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_real.csv"
python src\ingest.py
python run_demo.py
python -m streamlit run app.py
```

## 4. 安装依赖

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
pip install -r requirements.txt
```

## 5. 构建索引

```powershell
python src\ingest.py
```

## 6. 跑命令行 demo

```powershell
python run_demo.py
```

## 7. 启动 Streamlit demo

```powershell
python -m streamlit run app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 8. 运行评估

```powershell
python evaluate.py
```

评估结果保存到：

```text
reports/evaluation_results.csv
```

## 9. 启动 MCP server

```powershell
python mcp_server.py
```

当前暴露的工具：

```text
extract_preferences
search_restaurants
recommend_restaurants
evaluate_recommender
```

## 10. 推荐演示输入

```text
今晚想和朋友聚餐，人均 80 元以内，想吃亚洲菜，不要太辣，离学校近一点。
```

```text
想找一家适合约会的餐厅，预算 180 元以内，希望环境安静、有氛围。
```

```text
中午只有 30 分钟，预算 45 元，想吃健康一点，最好能快速出餐。
```

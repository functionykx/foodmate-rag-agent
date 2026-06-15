# 打工人生活助手使用说明

项目现在支持两类生活决策：

- 餐厅推荐：预算、菜系、场景、距离、辣度、优惠和出行时间。
- 租房推荐：月租、户型、面积、设施、验真状态，以及到公司或学校的真实通勤距离和时间。

## 1. 启动网页

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
$env:FOODMATE_RECALL_TOP_K="30"
python -m streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

餐厅示例：

```text
预算80元，和同事吃工作日晚餐，想吃辣一点的中餐
```

租房示例：

```text
想在大运附近整租一室一厅，月租3000元以内，要近地铁、精装修
```

带通勤条件的示例：

```text
预算4500元以内，至少两室，通勤到大运中心地铁站B口，骑行30分钟以内
```

信息不足时 Agent 会追问：

```text
用户：我想在大运附近租房
Agent：你的月租预算上限是多少？
用户：3500元
```

## 2. 启动 API

```powershell
python -m uvicorn api:app --reload --port 8000
```

通用 Agent 会自动路由：

```powershell
$body = @{ query = "租房预算4000元，要两室，近地铁" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/agent" -Method Post -ContentType "application/json" -Body $body
```

租房专用接口：

```powershell
$body = @{
    query = "月租4500元以内，至少两室"
    commute_destination = "大运中心地铁站B口"
    transport_mode = "riding"
    top_k = 5
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/rental/recommend" -Method Post -ContentType "application/json" -Body $body
```

餐厅专用接口仍为 `/recommend`。

## 3. 租房 RAG 链路

```text
用户输入
→ Supervisor 判断 rental intent
→ Rental Agent 抽取结构化偏好
→ 信息不足时追问月租预算
→ Query Rewrite
→ Hybrid Retrieval Top30
→ 可选 CrossEncoder 语义重排 Top30
→ 基础 Business Rerank Top10
   - 预算匹配
   - 户型匹配
   - 面积匹配
   - 设施匹配
   - 地铁与装修偏好
   - 朝向
   - 验真状态
→ 百度地图批量算路 Top10
→ 距离与时间参与最终重排 Top5
→ Critic 检查字段与结果
→ Top5 可解释推荐与风险提示
```

语义重排位于地图调用之前：它先判断 Top30 候选与需求是否相关，再让业务规则筛出 Top10。这样只对更有希望的 10 套调用百度算路，减少延迟和额度消耗。

启用 CrossEncoder：

```powershell
$env:FOODMATE_RENTAL_PIPELINE_MODE="hybrid+cross_encoder"
```

广告房源可以参与召回，但 `verified_score` 低于带 `SZ` 验真编号的房源。用户输入“只要真实验真房源”时，广告房源会进一步降权。

## 4. 补全租房坐标

26 套房源对应 12 个唯一小区。脚本对每个小区只查询一次，再将坐标回填给同小区全部房源：

```powershell
python scripts\enrich_rental_coordinates.py --limit 1
python scripts\enrich_rental_coordinates.py --retry-failed --delay 0.25
```

输出为 `data\rental_listings_cuhksz_geo.csv`，存在时 Rental Agent 自动优先加载。目前已完成 `26/26` 套坐标补全。

## 5. 运行测试

```powershell
python -m unittest discover -s tests -v
```

重点测试包括餐厅地图工具、租房文本解析、租房偏好抽取、Top30 召回、Supervisor 路由和多轮追问。

## 6. 数据限制

- 当前租房知识库只有大运附近的小型样本，推荐用于项目演示，不代表完整市场房源。
- 房源价格、在租状态和费用会变化，最终必须返回原平台再次核验。
- 通勤时间来自百度路线服务，会受交通方式、道路和实时状态影响，只用于筛选参考。

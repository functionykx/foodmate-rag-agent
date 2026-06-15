# 百度地图接入与使用流程

## 1. 功能说明

FoodMate 现在支持步行、骑行和驾车三种路线方式：

- 用户没有填写位置：使用知识库中的 `distance_km`，表示学校到餐厅的默认距离。
- 用户填写实际位置：调用百度地图解析起点，并按所选方式计算路线距离和预计时间。

租房链路为了控制算路额度，会先做语义和基础业务筛选，再对 Top10 调用地图：

```text
用户输入
→ 偏好与位置抽取
→ Hybrid/BGE Top30
→ CrossEncoder
→ 基础 Business Rerank Top10
→ 百度步行/骑行/驾车批量算路 Top10
→ 距离和时间重排
→ Top5
```

百度地图失败、AK 未配置或餐厅缺少坐标时，系统自动回退到学校距离，不会中断推荐。

## 2. 配置 AK

仅对当前 PowerShell 窗口生效：

```powershell
$env:BAIDU_MAP_AK="你的AK"
```

永久写入当前 Windows 用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable(
    "BAIDU_MAP_AK",
    "你的AK",
    "User"
)
```

永久配置后需要重新打开 PowerShell。检查是否存在，不要把 AK 打印到截图或提交到 GitHub：

```powershell
if ($env:BAIDU_MAP_AK) { "AK 已配置" } else { "AK 未配置" }
```

## 3. 给餐厅补齐百度坐标

进入项目目录：

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
```

先处理一条测试：

```powershell
python scripts\enrich_restaurant_coordinates.py --limit 1
```

确认生成的 `data\restaurants_cuhksz_geo.csv` 中包含：

```text
latitude
longitude
coordinate_type
baidu_uid
coordinate_source
coordinate_query
geocode_status
geocode_error
coordinate_updated_at
```

测试正常后补齐全部餐厅：

```powershell
python scripts\enrich_restaurant_coordinates.py
```

脚本每处理一家餐厅都会保存一次，关闭窗口后重新运行会自动从断点继续。失败项不会在普通续跑时反复消耗配额，需要重试时执行：

```powershell
python scripts\enrich_restaurant_coordinates.py --retry-failed
```

重新从原始CSV开始查询：

```powershell
python scripts\enrich_restaurant_coordinates.py --no-resume
```

脚本优先用餐厅名称进行地点检索，失败时使用文字地址进行地理编码。查询结果还会缓存到：

```text
vector_store\baidu_geocode_cache.json
```

重复运行时会跳过已有坐标，并复用缓存，减少百度接口调用。

## 4. 给租房补齐百度坐标

租房坐标按唯一小区补全，而不是逐套查询。26 套房源对应 12 个唯一小区，因此完整补全最多需要 12 次地点检索：

```powershell
python scripts\enrich_rental_coordinates.py --limit 1
python scripts\enrich_rental_coordinates.py --retry-failed --delay 0.25
```

脚本生成 `data\rental_listings_cuhksz_geo.csv`，并把同小区坐标回填到全部房源。Rental Agent 检测到该文件后会自动优先加载。

## 5. 使用带坐标的数据文件

当前 PowerShell 窗口配置：

```powershell
$env:FOODMATE_DATA_PATH="data\restaurants_cuhksz_geo.csv"
```

项目检测到该文件存在时会默认优先加载，因此通常不再需要手动设置；环境变量主要用于临时切换其他数据集。

检查：

```powershell
python -c "from src.utils import load_restaurants; d=load_restaurants(); print(d[['name','latitude','longitude']].head())"
```

## 6. 启动网页

```powershell
python -m streamlit run app.py
```

在侧边栏的“当前位置（选填）”中填写：

```text
大运中心地铁站C口
```

聊天框输入：

```text
预算80元，和同学吃晚餐，想吃辣一点
```

推荐表中应出现：

- `effective_distance_km`：用户位置到餐厅的路线距离。
- `estimated_duration_min`：所选方式的预计时间。
- `transport_mode`：`walking`、`riding` 或 `driving`。
- `distance_source`：`baidu_walking`、`baidu_riding` 或 `baidu_driving`。

如果侧边栏位置留空，则使用：

```text
distance_source=school_default
```

## 7. 自然语言输入位置

也可以直接在需求中写：

```text
我现在在大运中心地铁站C口，预算80元，想吃湘菜
```

支持的典型表达：

```text
我在大运天地
当前位置：深圳北理莫斯科大学
从大运中心地铁站C口出发
起点是龙城公园站A口
```

## 8. FastAPI 使用

启动：

```powershell
python -m uvicorn api:app --reload --port 8000
```

调用推荐接口：

```powershell
$body = @{
    query = "预算80元，和同学吃晚餐，想吃辣一点"
    user_location = "大运中心地铁站C口"
    transport_mode = "riding"
    pipeline_mode = "hybrid"
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/recommend" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

响应中的 `map_context` 会显示：

```json
{
  "enabled": true,
  "user_location": "大运中心地铁站C口",
  "transport_mode": "riding",
  "routed_candidates": 50,
  "fallback_candidates": 0
}
```

租房接口示例：

```powershell
$body = @{
    query = "预算4500元以内，至少两室"
    commute_destination = "大运中心地铁站B口"
    transport_mode = "riding"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/rental/recommend" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

默认 `FOODMATE_MAP_BATCH_SIZE=10` 时，Top10 使用一次批量矩阵请求，而不是逐套发起 10 次请求。

## 9. 常见问题

### 页面显示“百度地图未配置”

启动 Streamlit 的 PowerShell 没有读取到 `BAIDU_MAP_AK`。重新设置变量并重启 Streamlit。

### `fallback_candidates` 大于 0

部分餐厅没有经纬度。重新运行坐标补全脚本，并检查失败餐厅的名称和地址是否完整。

### 地点识别错误

把起点写得更完整，例如：

```text
深圳市龙岗区大运中心地铁站C口
```

### 请求超时或配额不足

系统会使用 `school_fallback`。可以在百度控制台检查配额，也可以调整：

```powershell
$env:FOODMATE_MAP_TIMEOUT="10"
$env:FOODMATE_MAP_BATCH_SIZE="10"
```

### 是否每次都查询餐厅坐标

不会。餐厅坐标由离线脚本写入 CSV，线上只解析用户起点并对候选餐厅算路。

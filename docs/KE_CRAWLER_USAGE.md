# 贝壳公开租房页面采集器使用说明

## 1. 使用边界

采集器只用于低频读取公开租房页面，默认单线程、每个详情页间隔5至8秒，并在出现验证码、访问异常或频控时停止。

请在运行前检查目标子域名的 `robots.txt` 和最新用户协议。不要使用脚本绕过登录、验证码、访问频控或其他技术限制，不采集经纪人姓名、电话号码、微信等个人信息。

## 2. 采集流程

```text
大运租房列表页
→ 收集普通房源详情链接
→ 保存链接清单
→ 逐个打开详情页
→ 保存可见原文JSONL
→ 解析验真编号、租金、户型、费用和设施
→ 按验真编号合并到租房知识库CSV
```

默认过滤方式是只接受形如 `/zufang/SZ数字.html` 的普通房源详情链接。广告公寓通常不符合该格式，不会进入默认清单。

## 3. 安装依赖

进入项目目录：

```powershell
cd "<PROJECT_ROOT>\rag_recommender_agent"
```

建议在当前项目的 Conda 环境中安装：

```powershell
pip install -r requirements-crawler.txt
python -m playwright install chromium
```

验证：

```powershell
python -c "from playwright.sync_api import sync_playwright; print('Playwright 已安装')"
```

## 4. 首次安全试跑

先只读取当前列表页并抓取3套详情：

```powershell
python scripts\crawl_ke_listings.py `
    --max-pages 1 `
    --max-details 3 `
    --manual-ready
```

脚本会启动一个独立的可见 Chromium，并使用：

```text
work\ke_browser_profile
```

保存该采集器自己的Cookie和浏览器状态，不会使用日常Chrome配置。

如果出现人工验证：

1. 在脚本打开的浏览器中手动完成验证。
2. 回到PowerShell按Enter。
3. 如果验证仍未解除，脚本会停止。

`--manual-ready` 会让脚本在列表页打开后固定等待。请先在Chromium中完成验证，确认页面已经显示“已为您找到XX套”和房源价格，再回到PowerShell按Enter。浏览器在等待期间不会自动关闭。

如果人工确认后仍未识别到链接，脚本会保存：

```text
data\raw\debug\ke_list_zero_links_时间.html
data\raw\debug\ke_list_zero_links_时间.png
```

这两个文件可用于检查网页是否仍是验证页，或者详情链接格式是否发生变化。

## 5. 输出文件

链接清单：

```text
data\raw\ke_listing_links.json
```

详情页原文和解析结果：

```text
data\raw\ke_listings.jsonl
```

结构化知识库：

```text
data\rental_listings_cuhksz.csv
```

原文JSONL用于解析规则回放。页面解析失败时，可以修改解析器后重新处理原文，不必再次访问网站。

重新解析本地原文：

```powershell
python scripts\reparse_ke_raw.py
```

## 6. 检查试跑结果

```powershell
Import-Csv "data\rental_listings_cuhksz.csv" |
    Select-Object listing_id,title,monthly_rent,area_sqm,maintenance_date,source_url |
    Format-Table -AutoSize
```

检查原始运行状态：

```powershell
Get-Content "data\raw\ke_listings.jsonl" |
    ForEach-Object { $_ | ConvertFrom-Json } |
    Select-Object url,status,fetched_at
```

## 7. 扩大到当前列表页

确认前3套字段正确后，可以抓当前页更多房源：

```powershell
python scripts\crawl_ke_listings.py `
    --use-saved-links `
    --max-details 20
```

`--use-saved-links` 会复用已经保存的链接清单。已成功抓取的URL会自动跳过。

## 8. 采集多页

重新扫描前3页列表，每次最多抓20套：

```powershell
python scripts\crawl_ke_listings.py `
    --max-pages 3 `
    --max-details 20 `
    --delay-min 6 `
    --delay-max 10
```

不要一开始直接抓全部66套。建议分批运行，并观察是否出现403、429、验证码或字段解析异常。

## 9. 无界面运行

完成首次浏览器验证并确认页面稳定后，才考虑：

```powershell
python scripts\crawl_ke_listings.py `
    --use-saved-links `
    --max-details 10 `
    --headless
```

无界面模式遇到验证时会直接停止，不能人工继续。

## 10. 自定义列表页

```powershell
python scripts\crawl_ke_listings.py `
    --list-url "https://sz.zu.ke.com/zufang/longgangqu/rs%E5%A4%A7%E8%BF%90/" `
    --max-pages 1 `
    --max-details 3
```

## 11. 续跑机制

脚本每完成一个详情页就立即追加JSONL，因此中途关闭后仍保留已经完成的数据。再次运行时：

```text
status=ok       自动跳过
status=error    可以重新访问
parse_failed    保留原文并允许后续修复解析器
```

CSV按 `listing_id` 去重；没有验真编号时退化为按 `source_url` 去重。

## 12. 运行解析测试

```powershell
$env:FOODMATE_USE_LLM="0"
python -m unittest tests.test_rental_parser -v
```

## 13. 常用参数

```text
--max-pages          最多扫描的列表页数
--max-details        本次最多访问的详情页数量
--delay-min/max      详情页之间随机等待秒数
--use-saved-links    复用已有链接清单
--headless           无界面模式
--profile-dir        独立浏览器配置目录
--raw-output         原始JSONL路径
--csv-output         租房知识库CSV路径
```

`--ignore-robots` 仅应在你确认已经获得网站授权时使用，不建议用于普通项目试跑。

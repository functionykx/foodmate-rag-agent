# 贝壳列表文本到Top10详情的离线流程

这套流程不自动访问贝壳网页。你使用正常浏览器查看公开页面，程序只处理复制到本地的文本。

## 第一步：复制列表页文本

在贝壳大运租房列表页按 `Ctrl+A`、`Ctrl+C`，将页面可见文本保存到：

```text
data\manual_ke\list_page.txt
```

PowerShell可以先创建文件：

```powershell
New-Item -ItemType File -Force "data\manual_ke\list_page.txt"
notepad "data\manual_ke\list_page.txt"
```

在记事本中粘贴并保存。不要删除“标题、小区/面积/户型、标签、维护时间、价格”等行。

## 第二步：解析列表候选

```powershell
python scripts\import_ke_list_text.py
```

输出：

```text
data\rental_candidates_cuhksz.csv
```

列表候选包含：

- 临时 `candidate_id`
- 标题和小区
- 月租区间
- 面积区间
- 户型和朝向
- 标签和维护日期
- 是否广告

列表页没有验真编号、押金和完整设施，因此这些数据暂时不能进入正式推荐知识库。

## 第三步：生成Top10待补详情清单

例如预算3500元、希望两居整租：

```powershell
python scripts\select_rental_top10.py `
    --budget 3500 `
    --bedrooms 2 `
    --rental-type "整租" `
    --keywords "近地铁,精装,随时看房" `
    --top-k 10
```

输出：

```text
data\rental_top10_to_enrich.csv
```

当前预选分为：

```text
50% 预算匹配
25% 户型匹配
15% 标签匹配
10% 维护新鲜度
```

这一步是轻量级候选预选，还不是完整的 Rental Agent。后续 Rental Agent 会在补齐详情和通勤时间后重新排序。

## 第四步：人工打开Top10详情页

打开清单：

```powershell
Import-Csv "data\rental_top10_to_enrich.csv" |
    Select-Object candidate_id,title,monthly_rent_min,detail_text_file |
    Format-Table -AutoSize
```

对每条房源执行：

1. 在正常浏览器列表页点击对应房源。
2. 复制详情页全部可见文本。
3. 打开清单中对应的 `detail_text_file`。
4. 粘贴并保存。
5. 可选：在CSV的 `detail_url` 列填入详情页URL。

例如：

```powershell
notepad "data\manual_ke\details\LIST-XXXXXXXXXXXX.txt"
```

详情文本必须尽量包含：

```text
房源验真编号
租金
租赁方式
房屋类型
面积、朝向、楼层
入住、看房、租期
电梯、水电、燃气
配套设施
付款方式和押金
维护日期
```

## 第五步：导入完整详情

```powershell
python scripts\import_ke_detail_texts.py
```

程序会：

1. 读取Top10清单。
2. 找到对应TXT详情文本。
3. 提取验真编号和完整字段。
4. 按验真编号去重。
5. 合并到正式知识库：

```text
data\rental_listings_cuhksz.csv
```

缺少详情文件的候选会显示 `[缺失]`，不会产生虚假数据。

## 第六步：检查正式知识库

```powershell
Import-Csv "data\rental_listings_cuhksz.csv" |
    Select-Object listing_id,title,monthly_rent,deposit,area_sqm,elevator,maintenance_date |
    Format-Table -AutoSize
```

## 第七步：补坐标和通勤

完整房源进入正式知识库后，可以用小区名称调用现有 `BaiduMapTool` 补经纬度，再计算房源到学校或公司的步行、骑行和驾车时间。

推荐流程最终是：

```text
列表页候选
→ 结构化预选Top10
→ 人工补详情
→ 完整知识库
→ Rental Retrieval
→ 百度通勤时间
→ Rental Business Rerank
→ Rental Critic
→ Top5房源
```

## 测试解析器

```powershell
python -m unittest tests.test_rental_parser -v
```

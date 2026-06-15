# 租房知识库字段说明

租房数据保存在 `data/rental_listings_cuhksz.csv`，与餐厅知识库分开管理，避免餐饮预算、场景等字段污染租房上下文。

## 核心字段

- `listing_id`：贝壳房源验真编号，作为房源唯一标识。
- `monthly_rent`、`deposit`、`service_fee`、`agency_fee`：租金及费用信息。
- `rental_type`、`bedrooms`、`living_rooms`、`bathrooms`：租赁方式和户型。
- `area_sqm`、`decoration`、`orientation`、`floor_level`：房屋条件。
- `move_in_date`、`lease_term`、`viewing`：入住、租期与看房要求。
- `facilities`：房源明确提供的设施，不根据常识补全。
- `verification_status`：房源验真状态。
- `publisher_compliance`：发布人或发布机构合规信息。
- `maintenance_date`：房源最近维护日期，用于时效衰减和过期检查。
- `availability_status`：当前是否在租，正式使用时需要定期更新。
- `latitude`、`longitude`：后续使用百度地图补齐，用于计算通勤距离和时间。

## 数据原则

1. 不保存经纪人姓名、个人电话等推荐无关信息。
2. “暂无数据”和“需咨询”不能推断为免费或不存在。
3. 房源会快速失效，推荐前必须检查 `maintenance_date` 和 `availability_status`。
4. 未登记或未合规备案的发布信息应进入风险提示，不应被模型隐藏。
5. 当前数据来自用户提供的网页文本，缺少具体房源链接时 `source_url` 留空。

## 后续补充坐标

可以复用项目中的 `BaiduMapTool`，使用小区名称和深圳市龙岗区作为检索条件，补充统一的 `bd09ll` 坐标。多个同小区房源可以共享小区级坐标，但需要明确其精度为小区级而不是楼栋级。

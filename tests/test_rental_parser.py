from __future__ import annotations

import unittest
from datetime import datetime

from src.rental_parser import parse_ke_list_page, parse_ke_listing, valid_listing


SAMPLE = """
整租·阳光天健城 1室0厅 东南
房源维护时间：2026-06-14
营业执照
房源验真编号：SZ1913450501664407552
2000元/月 (月付价)
贝壳省心租 近地铁 精装 押一付一 随时看房
租赁方式：整租
房屋类型：1室0厅1卫 36.00㎡ 精装修
朝向楼层：东南 低楼层/27层
基本信息
面积：36.00㎡ 朝向：东南 维护：今天 入住：2026-07-01 楼层：低楼层/27层 电梯：有 车位：暂无数据 用水：民水 用电：民电 燃气：有 采暖：暂无数据
租期：暂无数据
看房：随时可看
配套设施
洗衣机
空调
衣柜
电视 无
冰箱
热水器
床
暖气 无
宽带 无
天然气
付款方式
租金 (元/月)
押金 (元)
服务费 (元/年)
中介费 (元)
月付
2000
2000
需咨询
0
"""

LIST_SAMPLE = """
整租·阳光天健城 1室1厅 东/东南_阳光天健城租房
整租·阳光天健城 1室1厅 东/东南
龙岗区-大运新城-阳光天健城 / 44.32㎡ /东 东南 / 1室1厅1卫
自营 精装 押一付一 随时看房
贝壳优选
2700 元/月
整租·森雅谷 2室1厅 东南_森雅谷租房
整租·森雅谷 2室1厅 东南
必看好房
龙岗区-大运新城-森雅谷 / 60.98㎡ /东南 / 2室1厅1卫
近地铁 精装 随时看房
2天前维护
3500 元/月
"""


class RentalParserTest(unittest.TestCase):
    def test_parse_list_page(self):
        records = parse_ke_list_page(
            LIST_SAMPLE,
            datetime.fromisoformat("2026-06-14T12:00:00+08:00"),
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["community"], "阳光天健城")
        self.assertEqual(records[0]["monthly_rent_min"], 2700)
        self.assertEqual(records[0]["bedrooms"], 1)
        self.assertEqual(records[1]["maintenance_date"], "2026-06-12")

    def test_parse_listing(self):
        record = parse_ke_listing(
            SAMPLE,
            "https://sz.zu.ke.com/zufang/SZ1913450501664407552.html",
            "龙岗区-大运新城-阳光天健城 / 36.00㎡ /东南 / 1室0厅1卫",
            datetime.fromisoformat("2026-06-14T12:00:00+08:00"),
        )
        self.assertTrue(valid_listing(record))
        self.assertEqual(record["listing_id"], "SZ1913450501664407552")
        self.assertEqual(record["community"], "阳光天健城")
        self.assertEqual(record["monthly_rent"], 2000)
        self.assertEqual(record["bedrooms"], 1)
        self.assertEqual(record["area_sqm"], 36.0)
        self.assertEqual(record["elevator"], "是")
        self.assertEqual(record["maintenance_date"], "2026-06-14")
        self.assertNotIn("电视", record["facilities"])
        self.assertNotIn("宽带", record["facilities"])


if __name__ == "__main__":
    unittest.main()

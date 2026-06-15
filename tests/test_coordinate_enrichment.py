from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.enrich_restaurant_coordinates import enrich
from scripts.enrich_rental_coordinates import enrich as enrich_rentals
from src.tools.baidu_map import BaiduMapError


class FakeMapTool:
    def __init__(self):
        self.queries = []

    def available(self):
        return True

    def search_place(self, name, region=None):
        self.queries.append(("place", name, region))
        if "失败店" in name:
            raise BaiduMapError("未找到地点")
        return {
            "latitude": 22.70242,
            "longitude": 114.226559,
            "coordinate_type": "bd09ll",
            "baidu_uid": "test-uid",
            "source": "baidu_place_search",
        }

    def geocode(self, address, city=None):
        self.queries.append(("geocode", address, city))
        if "失败店地址" in address:
            raise BaiduMapError("地理编码失败")
        return {
            "latitude": 22.70001,
            "longitude": 114.22001,
            "coordinate_type": "bd09ll",
            "source": "baidu_geocoding",
        }


class CoordinateEnrichmentTest(unittest.TestCase):
    def test_enrich_and_record_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "restaurants.csv"
            output_path = root / "restaurants_geo.csv"
            pd.DataFrame(
                [
                    {"id": 1, "name": "测试餐厅", "location": "测试路1号"},
                    {"id": 2, "name": "失败店", "location": "失败店地址"},
                ]
            ).to_csv(input_path, index=False, encoding="utf-8")

            result = enrich(
                input_path=input_path,
                output_path=output_path,
                delay=0,
                tool=FakeMapTool(),
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result.loc[0, "geocode_status"], "success")
            self.assertEqual(result.loc[0, "baidu_uid"], "test-uid")
            self.assertAlmostEqual(float(result.loc[0, "latitude"]), 22.70242)
            self.assertEqual(result.loc[1, "geocode_status"], "failed")
            self.assertTrue(str(result.loc[1, "geocode_error"]))

    def test_resume_skips_existing_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "restaurants.csv"
            output_path = root / "restaurants_geo.csv"
            pd.DataFrame([{"id": 1, "name": "测试餐厅", "location": "测试路1号"}]).to_csv(
                input_path, index=False, encoding="utf-8"
            )
            tool = FakeMapTool()
            enrich(input_path, output_path, delay=0, tool=tool)
            query_count = len(tool.queries)
            enrich(input_path, output_path, delay=0, tool=tool)
            self.assertEqual(len(tool.queries), query_count)

    def test_rental_enrichment_queries_each_community_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "rentals.csv"
            output_path = root / "rentals_geo.csv"
            pd.DataFrame(
                [
                    {"listing_id": "a", "community": "测试小区A"},
                    {"listing_id": "b", "community": "测试小区A"},
                    {"listing_id": "c", "community": "测试小区B"},
                ]
            ).to_csv(input_path, index=False, encoding="utf-8")
            tool = FakeMapTool()

            result = enrich_rentals(
                input_path=input_path,
                output_path=output_path,
                delay=0,
                tool=tool,
            )

            place_queries = [item for item in tool.queries if item[0] == "place"]
            self.assertEqual(len(place_queries), 2)
            self.assertEqual(result["latitude"].notna().sum(), 3)
            same_community = result[result["community"] == "测试小区A"]
            self.assertEqual(same_community["latitude"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()

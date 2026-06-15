from __future__ import annotations

import os
import unittest

from src.agent import FoodMateAgent, extract_transport_mode, extract_user_location
from src.retriever import RestaurantRetriever
from src.tools.baidu_map import BaiduMapTool, ROUTE_MATRIX_URLS
from src.utils import load_restaurants


class FakeMapTool:
    def available(self) -> bool:
        return True

    def update_candidate_distances(self, user_location, candidates, transport_mode="walking"):
        result = candidates.copy()
        result["effective_distance_km"] = 0.8
        result["estimated_duration_min"] = 11.0
        result["transport_mode"] = transport_mode
        result["distance_source"] = f"baidu_{transport_mode}"
        return result, {
            "enabled": True,
            "user_location": user_location,
            "transport_mode": transport_mode,
            "routed_candidates": len(result),
            "fallback_candidates": 0,
        }


class FakeBaiduMapTool(BaiduMapTool):
    def __init__(self):
        super().__init__(ak="test-ak")
        self.requested_urls = []

    def _request(self, url, params):
        self.requested_urls.append(url)
        count = max(
            len(params["origins"].split("|")),
            len(params["destinations"].split("|")),
        )
        return {
            "status": 0,
            "result": [
                {"distance": {"value": 1200}, "duration": {"value": 600}}
                for _ in range(count)
            ],
        }


class BaiduMapIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FOODMATE_USE_LLM"] = "0"

    def test_02_extract_location(self):
        self.assertEqual(
            extract_user_location("预算80元，我现在在大运中心地铁站C口，想吃湘菜"),
            "大运中心地铁站C口",
        )
        self.assertEqual(
            extract_user_location("从深圳北理莫斯科大学出发，想吃晚餐"),
            "深圳北理莫斯科大学",
        )
        self.assertEqual(extract_transport_mode("我想骑车过去"), "riding")
        self.assertEqual(extract_transport_mode("准备打车过去"), "driving")
        self.assertEqual(extract_transport_mode("走路过去"), "walking")

    def test_01_map_distance_runs_before_business_rerank(self):
        data = load_restaurants("data/restaurants_cuhksz.csv")
        data["latitude"] = 22.69
        data["longitude"] = 114.21
        retriever = RestaurantRetriever(data, mode="hybrid")
        agent = FoodMateAgent(retriever=retriever, pipeline_mode="hybrid", map_tool=FakeMapTool())
        result = agent.handle("预算80元，和同学吃晚餐。当前位置：大运中心地铁站C口。骑行")

        self.assertEqual(result["type"], "recommendation")
        self.assertTrue(result["map_context"]["enabled"])
        self.assertTrue((result["recommendations"]["distance_source"] == "baidu_riding").all())
        self.assertTrue((result["recommendations"]["transport_mode"] == "riding").all())
        self.assertIn("travel_time_score", result["recommendations"].columns)
        actions = [item["action"] for item in result["actions"]]
        self.assertLess(actions.index("baidu_riding_route"), actions.index("business_rerank"))

    def test_03_three_route_matrix_modes(self):
        tool = FakeBaiduMapTool()
        for mode in ["walking", "riding", "driving"]:
            routes = tool.route_matrix((22.69, 114.21), [(22.70, 114.22)], mode)
            self.assertEqual(tool.requested_urls[-1], ROUTE_MATRIX_URLS[mode])
            self.assertEqual(routes[0]["distance_km"], 1.2)
            self.assertEqual(routes[0]["duration_min"], 10.0)

    def test_04_many_origins_to_one_destination(self):
        tool = FakeBaiduMapTool()
        routes = tool.route_many_origins_to_destination(
            [(22.69, 114.21), (22.70, 114.22)],
            (22.71, 114.23),
            "riding",
        )
        self.assertEqual(len(tool.requested_urls), 1)
        self.assertEqual(tool.requested_urls[0], ROUTE_MATRIX_URLS["riding"])
        self.assertEqual(len(routes), 2)


if __name__ == "__main__":
    unittest.main()

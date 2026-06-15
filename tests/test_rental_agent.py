from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.multi_agent import SupervisorAgent
from src.rental_agent import RentalAgent, RentalRetriever, extract_rental_preferences


class FakeRentalMapTool:
    def __init__(self):
        self.route_calls = 0
        self.origin_counts = []

    def available(self):
        return True

    def resolve_location(self, destination):
        return {
            "name": destination,
            "latitude": 22.70,
            "longitude": 114.22,
            "coordinate_type": "bd09ll",
        }

    def route_many_origins_to_destination(self, origins, destination, transport_mode="walking"):
        self.route_calls += 1
        self.origin_counts.append(len(origins))
        return [
            {"distance_km": 0.5 + index * 0.1, "duration_min": 5.0 + index}
            for index in range(len(origins))
        ]


class RentalAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FOODMATE_USE_LLM"] = "0"
        os.environ["FOODMATE_RECALL_TOP_K"] = "30"

    def test_extract_preferences(self):
        prefs = extract_rental_preferences("预算3000元以内，整租一室一厅，近地铁，要空调和天然气")
        self.assertEqual(prefs["max_rent"], 3000)
        self.assertEqual(prefs["bedrooms"], 1)
        self.assertEqual(prefs["rental_type"], "整租")
        self.assertTrue(prefs["near_metro"])
        self.assertIn("空调", prefs["required_facilities"])
        self.assertIn("天然气", prefs["required_facilities"])

    def test_rental_recommendation_pipeline(self):
        agent = RentalAgent()
        result = agent.handle("大运附近租房，月租4000元以内，至少两室，近地铁")
        self.assertEqual(result["type"], "rental_recommendation")
        self.assertLessEqual(len(result["recommendations"]), 5)
        self.assertIn("verified_score", result["recommendations"].columns)
        retrieval = [item for item in result["actions"] if item["action"] == "hybrid_retrieval"]
        self.assertEqual(retrieval[0]["detail"]["top_k"], 30)

    def test_minimum_bedrooms(self):
        prefs = extract_rental_preferences("月租5000元以内，至少两室")
        self.assertEqual(prefs["min_bedrooms"], 2)
        self.assertNotIn("bedrooms", prefs)

    def test_simple_budget_followup_is_recognized(self):
        prefs = extract_rental_preferences("预算2000")
        self.assertEqual(prefs["max_rent"], 2000)

    def test_two_person_shared_rental_prefers_two_bedrooms(self):
        supervisor = SupervisorAgent()
        first = supervisor.run("和朋友两人合租，离港中深近些")
        self.assertEqual(first["type"], "followup")

        second = supervisor.run("预算4000")
        recs = second["recommendations"]

        self.assertEqual(second["type"], "rental_recommendation")
        self.assertEqual(second["preferences"]["preferred_min_bedrooms"], 2)
        self.assertEqual(second["preferences"]["commute_destination"], "香港中文大学（深圳）")
        self.assertTrue((recs["monthly_rent"].astype(float) <= 4000).all())
        self.assertGreaterEqual(int(recs.iloc[0]["bedrooms"]), 2)

    def test_budget_remains_hard_constraint_when_fewer_than_top_k(self):
        supervisor = SupervisorAgent()
        supervisor.run("和朋友两人合租")
        result = supervisor.run("预算2000")
        recs = result["recommendations"]

        self.assertEqual(result["preferences"]["max_rent"], 2000)
        self.assertTrue((recs["monthly_rent"].astype(float) <= 2000).all())

    def test_supervisor_routes_rental(self):
        supervisor = SupervisorAgent()
        result = supervisor.run("想在大运附近租房，预算3500元，要一室一厅")
        self.assertEqual(result["supervisor_intent"], "rental")
        self.assertEqual(result["active_agent"], "rental_agent")

    def test_followup_keeps_rental_context(self):
        supervisor = SupervisorAgent()
        first = supervisor.run("我想在大运附近租房")
        self.assertEqual(first["type"], "followup")
        second = supervisor.run("3500元")
        self.assertEqual(second["supervisor_intent"], "rental")
        self.assertEqual(second["type"], "rental_recommendation")

    def test_followup_keeps_rental_context_for_per_person_budget(self):
        supervisor = SupervisorAgent()
        first = supervisor.run("要找租房，两人合租")
        self.assertEqual(first["type"], "followup")
        self.assertEqual(first["supervisor_intent"], "rental")

        second = supervisor.run("预算人均2000")

        self.assertEqual(second["supervisor_intent"], "rental")
        self.assertEqual(second["active_agent"], "rental_agent")
        self.assertEqual(second["type"], "rental_recommendation")
        self.assertEqual(second["preferences"]["occupant_count"], 2)
        self.assertEqual(second["preferences"]["per_person_rent"], 2000)
        self.assertEqual(second["preferences"]["max_rent"], 4000)

    def test_explicit_food_request_can_leave_rental_followup(self):
        supervisor = SupervisorAgent()
        supervisor.run("我想租房")
        result = supervisor.run("先不租了，我想吃火锅")
        self.assertEqual(result["supervisor_intent"], "recommendation")

    def test_commute_pipeline_routes_only_business_top10(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "rentals.csv"
            rows = []
            for index in range(26):
                rows.append({
                    "listing_id": f"r{index}",
                    "title": f"测试房源{index}",
                    "community": f"测试小区{index % 12}",
                    "city": "深圳市",
                    "district": "龙岗区",
                    "monthly_rent": 2500 + index * 50,
                    "payment_method": "月付",
                    "deposit": 2500,
                    "service_fee": 0,
                    "agency_fee": 0,
                    "rental_type": "整租",
                    "bedrooms": 2,
                    "living_rooms": 1,
                    "bathrooms": 1,
                    "area_sqm": 60 + index,
                    "decoration": "精装修",
                    "orientation": "南",
                    "floor_level": "中楼层",
                    "total_floors": 30,
                    "elevator": "有",
                    "parking": "暂无数据",
                    "water_type": "民水",
                    "electricity_type": "民电",
                    "gas": "有",
                    "heating": "无",
                    "broadband": "有",
                    "move_in_date": "随时入住",
                    "lease_term": "1年以内",
                    "viewing": "需预约",
                    "facilities": "洗衣机;空调;冰箱;床",
                    "tags": "近地铁;精装",
                    "verification_status": "贝壳验真",
                    "publisher_compliance": "已备案",
                    "maintenance_date": "2026-06-15",
                    "availability_status": "在租",
                    "latitude": 22.68 + index * 0.001,
                    "longitude": 114.20 + index * 0.001,
                    "coordinate_type": "bd09ll",
                    "source": "test",
                    "source_url": "",
                })
            pd.DataFrame(rows).to_csv(data_path, index=False, encoding="utf-8")
            map_tool = FakeRentalMapTool()
            agent = RentalAgent(retriever=RentalRetriever(data_path), map_tool=map_tool)

            result = agent.handle(
                "大运附近租房，预算4500元以内，至少两室，通勤目的地大运中心地铁站B口，骑行30分钟以内"
            )

            self.assertEqual(result["type"], "rental_recommendation")
            self.assertEqual(map_tool.route_calls, 1)
            self.assertEqual(map_tool.origin_counts, [10])
            self.assertEqual(result["map_context"]["routed_candidates"], 10)
            self.assertEqual(len(result["recommendations"]), 5)
            self.assertTrue(result["recommendations"]["commute_duration_min"].notna().all())
            actions = [item["action"] for item in result["actions"]]
            self.assertLess(actions.index("hybrid_retrieval"), actions.index("business_prerank"))
            self.assertLess(actions.index("business_prerank"), actions.index("baidu_riding_route_top10"))
            self.assertLess(actions.index("baidu_riding_route_top10"), actions.index("commute_business_rerank"))


if __name__ == "__main__":
    unittest.main()

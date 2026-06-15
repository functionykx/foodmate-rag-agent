from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.agent import FoodMateAgent, extract_preferences
from src.multi_agent import MultiAgentState, RecommendationAgent


class CriticFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["FOODMATE_USE_LLM"] = "0"
        os.environ["FOODMATE_PIPELINE_MODE"] = "hybrid"
        os.environ["FOODMATE_RECALL_TOP_K"] = "30"
        project_root = Path(__file__).resolve().parents[1]
        os.environ["FOODMATE_DATA_PATH"] = str(project_root / "data" / "restaurants_cuhksz_geo.csv")

    def test_internal_critic_refills_from_candidate_pool(self):
        agent = FoodMateAgent(pipeline_mode="hybrid")
        result = agent.handle("想吃客家菜，预算120元，朋友聚餐")
        recs = result["recommendations"]
        self.assertFalse(recs.empty)
        self.assertTrue((recs["cuisine_score"] >= 0.9).all())
        self.assertTrue(result["critic_report"]["passed"])
        repair_actions = [item for item in result["actions"] if item["action"] == "repair_filter_and_refill"]
        if repair_actions:
            self.assertTrue(repair_actions[0]["detail"]["allow_less_than_top5"])

    def test_supervisor_feedback_changes_retry_strategy(self):
        agent = RecommendationAgent(pipeline_mode="hybrid")
        agent.apply_critic_feedback(
            {
                "passed": False,
                "issues": [
                    {"type": "cuisine_mismatch", "name": "吴庄(龙岗大运天地店)", "penalty": 0.18}
                ],
            }
        )
        state = MultiAgentState(retry_count=1)
        result = agent.run("想吃客家菜，预算120元，朋友聚餐", state)
        recs = result["recommendations"]
        self.assertTrue((recs["cuisine_score"] >= 0.9).all())
        actions = [item for item in result["actions"] if item["action"] == "critic_feedback_applied"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["detail"]["strategy"], "hard_filter_and_refill_from_top30")
        self.assertNotIn("吴庄(龙岗大运天地店)", set(recs["name"]))

    def test_budget_repair_keeps_explicit_cuisine_constraint(self):
        agent = FoodMateAgent(pipeline_mode="hybrid")
        result = agent.handle("周三晚饭自己一个人吃，要火锅，预算60元")
        recs = result["recommendations"]

        self.assertFalse(recs.empty)
        self.assertLessEqual(len(recs), 5)
        self.assertTrue((recs["cuisine_score"] >= 0.9).all())
        self.assertTrue((recs["price_per_person"].astype(float) <= 69.0).all())
        self.assertNotIn("文通冰室(大运天地店)", set(recs["name"]))
        self.assertNotIn("Potato Corner(深圳大运天地店)", set(recs["name"]))

    def test_rule_extractor_keeps_hotpot_as_specific_cuisine(self):
        preferences = extract_preferences("周三晚饭自己一个人吃，要火锅，预算60元")
        self.assertEqual(preferences["cuisine"], "火锅")


if __name__ == "__main__":
    unittest.main()

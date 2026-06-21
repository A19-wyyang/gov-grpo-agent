import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.decision import evaluate_final_decision, infer_decision_label


class DecisionTests(unittest.TestCase):
    def test_infer_decision_label_detects_material_missing(self):
        label = infer_decision_label("材料不完整，需补充租赁备案证明后申请。")

        self.assertEqual(label, "material_missing")

    def test_infer_decision_label_detects_ineligible(self):
        label = infer_decision_label("不符合当前办理条件，暂不能提交申请。")

        self.assertEqual(label, "ineligible")

    def test_infer_decision_label_detects_success(self):
        label = infer_decision_label("符合办理条件，材料齐全，可提交申请。")

        self.assertEqual(label, "success")

    def test_evaluate_final_decision_accepts_semantic_success_answer(self):
        case = build_case("housing_fund", 1, "simple_success")

        result = evaluate_final_decision(
            case,
            "您符合租房提取公积金条件，材料齐全，可向杭州公积金中心提交申请。",
        )

        self.assertTrue(result["correct"])
        self.assertEqual(result["expected_label"], "success")
        self.assertEqual(result["actual_label"], "success")

    def test_evaluate_final_decision_requires_missing_material_name(self):
        case = build_case("housing_fund", 2, "material_missing")

        good = evaluate_final_decision(case, "材料不完整，需补充租赁备案证明后申请。")
        bad = evaluate_final_decision(case, "材料不完整，需补充材料后申请。")

        self.assertTrue(good["correct"])
        self.assertFalse(bad["correct"])
        self.assertIn("missing_material:not_mentioned:租赁备案证明", bad["failure_reasons"])

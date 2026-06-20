import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.tools import EligibilityCheck, MaterialCheck, PolicySearch


class ToolTests(unittest.TestCase):
    def test_policy_search_returns_conditions_and_required_materials(self):
        case = build_case("housing_fund", 1, "simple_success")
        result = PolicySearch().run(
            {
                "service_item": case["service_item"],
                "city": case["user_profile"]["city"],
                "query": "申请条件 材料",
            },
            case,
        )

        self.assertIn("正常缴存住房公积金", result["conditions"])
        self.assertIn("租赁备案证明", result["required_materials"])

    def test_eligibility_check_reports_failed_conditions(self):
        case = build_case("housing_fund", 2, "ineligible")
        result = EligibilityCheck().run({}, case)

        self.assertFalse(result["eligible"])
        self.assertIn("continuous_payment_months", result["failed_conditions"])
        self.assertEqual(result["uncertain_slots"], [])

    def test_material_check_reports_missing_materials(self):
        case = build_case("housing_fund", 3, "material_missing")
        result = MaterialCheck().run({}, case)

        self.assertFalse(result["complete"])
        self.assertIn("租赁备案证明", result["missing"])
        self.assertIn("身份证", result["provided"])

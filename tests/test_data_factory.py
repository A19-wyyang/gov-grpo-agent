from collections import Counter
import unittest

from gov_grpo_agent.data import build_mvp_cases, load_policy_catalog
from gov_grpo_agent.schemas import validate_case


class DataFactoryTests(unittest.TestCase):
    def test_build_mvp_cases_matches_required_distribution(self):
        cases = build_mvp_cases()

        self.assertEqual(len(cases), 200)
        self.assertEqual(
            Counter(case["path_type"] for case in cases),
            {
                "simple_success": 60,
                "missing_information": 50,
                "ineligible": 40,
                "material_missing": 30,
                "complex_mixed": 20,
            },
        )
        self.assertEqual(len({case["service_item"] for case in cases}), 5)
        self.assertTrue(all(validate_case(case) for case in cases))

    def test_case_hidden_truth_contains_required_training_signals(self):
        case = build_mvp_cases()[0]
        hidden_truth = case["hidden_truth"]

        self.assertIn("required_tools", hidden_truth)
        self.assertIn("missing_slots", hidden_truth)
        self.assertIn("required_materials", hidden_truth)
        self.assertIn("missing_materials", hidden_truth)
        self.assertIn("final_decision", hidden_truth)

    def test_policy_catalog_contains_five_mvp_services(self):
        catalog = load_policy_catalog()

        self.assertEqual(len(catalog), 5)
        self.assertIn("租房提取公积金", catalog)
        self.assertEqual(
            catalog["租房提取公积金"]["tools"],
            ["Policy_Search", "Eligibility_Check", "Material_Check"],
        )

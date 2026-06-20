import unittest

from gov_grpo_agent.normalization import normalize_action_arguments, normalize_service_item


class NormalizationTests(unittest.TestCase):
    def test_normalize_service_item_maps_common_housing_fund_aliases(self):
        aliases = ["住房公积金提取", "公积金租房提取", "租房提取", "租房提取公积金"]

        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertEqual(normalize_service_item(alias), "租房提取公积金")

    def test_normalize_action_arguments_updates_service_item_without_mutating_input(self):
        action = {
            "action": "Policy_Search",
            "arguments": {
                "service_item": "住房公积金提取",
                "city": "杭州",
                "query": "租房提取条件 材料",
            },
        }

        normalized = normalize_action_arguments(action)

        self.assertEqual(normalized["arguments"]["service_item"], "租房提取公积金")
        self.assertEqual(action["arguments"]["service_item"], "住房公积金提取")

    def test_unknown_service_item_is_preserved(self):
        self.assertEqual(normalize_service_item("未知事项"), "未知事项")

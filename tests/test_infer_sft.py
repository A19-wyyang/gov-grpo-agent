import json
import unittest

from gov_grpo_agent.infer_sft import (
    build_inference_prompt,
    parse_action_from_generation,
)


class InferSftTests(unittest.TestCase):
    def test_build_inference_prompt_uses_chatml_roles(self):
        prompt = build_inference_prompt("我想提取公积金交房租，应该怎么办？")

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_parse_action_from_generation_accepts_plain_json(self):
        action = parse_action_from_generation(
            json.dumps(
                {
                    "action": "Policy_Search",
                    "arguments": {"service_item": "租房提取公积金"},
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(action["action"], "Policy_Search")
        self.assertEqual(action["arguments"]["service_item"], "租房提取公积金")

    def test_parse_action_from_generation_extracts_json_from_extra_text(self):
        action = parse_action_from_generation(
            '下一步应调用工具：{"action": "Ask_User", "arguments": {"slots": ["city"]}}<|im_end|>'
        )

        self.assertEqual(action["action"], "Ask_User")
        self.assertEqual(action["arguments"]["slots"], ["city"])

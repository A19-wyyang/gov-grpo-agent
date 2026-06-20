import json
import unittest
from collections import Counter

from gov_grpo_agent.data import build_mvp_cases
from gov_grpo_agent.sft import build_sft_samples


class SftTests(unittest.TestCase):
    def test_build_sft_samples_reaches_target_count_and_chatml_shape(self):
        samples = build_sft_samples(build_mvp_cases(limit=20), target_count=2000)

        self.assertEqual(len(samples), 2000)
        first = samples[0]
        self.assertIn("sample_id", first)
        self.assertIn("sample_type", first)
        self.assertEqual([message["role"] for message in first["messages"]], ["system", "user", "assistant"])
        assistant_payload = json.loads(first["messages"][-1]["content"])
        self.assertIn("action", assistant_payload)
        self.assertIn("arguments", assistant_payload)

    def test_build_sft_samples_covers_required_sample_types(self):
        samples = build_sft_samples(build_mvp_cases(limit=20), target_count=2000)
        counts = Counter(sample["sample_type"] for sample in samples)

        self.assertEqual(
            set(counts),
            {
                "single_action",
                "multi_tool",
                "missing_slot_question",
                "material_missing_answer",
                "ineligible_refusal",
                "complex_multiturn",
            },
        )
        self.assertGreaterEqual(counts["single_action"], 500)
        self.assertGreaterEqual(counts["multi_tool"], 500)

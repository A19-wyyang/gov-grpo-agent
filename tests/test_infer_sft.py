import json
import unittest

from gov_grpo_agent.infer_sft import (
    SftActionGenerator,
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

    def test_sft_action_generator_passes_sampling_parameters_to_model_generate(self):
        generator = SftActionGenerator.__new__(SftActionGenerator)
        generator.max_new_tokens = 128
        generator.do_sample = True
        generator.temperature = 0.9
        generator.top_p = 0.85
        generator.tokenizer = FakeTokenizer()
        generator.model = FakeModel()
        generator.torch = FakeTorch()

        action = generator.generate("下一步做什么？")

        self.assertEqual(action["action"], "Policy_Search")
        self.assertTrue(generator.model.generate_kwargs["do_sample"])
        self.assertEqual(generator.model.generate_kwargs["temperature"], 0.9)
        self.assertEqual(generator.model.generate_kwargs["top_p"], 0.85)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeInputIds:
    shape = (1, 2)


class FakeGeneratedRow:
    def __getitem__(self, item):
        return [3, 4]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors):
        return FakeBatch({"input_ids": FakeInputIds()})

    def decode(self, generated_ids, skip_special_tokens=False):
        return '{"action": "Policy_Search", "arguments": {"service_item": "租房提取公积金"}}'


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [FakeGeneratedRow()]


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    def no_grad(self):
        return FakeNoGrad()

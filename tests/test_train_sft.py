import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.train_sft import (
    SftTrainingConfig,
    format_chatml_sample,
    load_sft_jsonl,
)


class TrainSftTests(unittest.TestCase):
    def test_default_training_config_targets_qwen3_8b(self):
        config = SftTrainingConfig()

        self.assertEqual(config.model_name_or_path, "Qwen/Qwen3-8B")
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.lora_rank, 16)
        self.assertTrue(config.load_in_4bit)

    def test_format_chatml_sample_contains_roles_and_json_action(self):
        sample = {
            "messages": [
                {"role": "system", "content": "系统提示"},
                {"role": "user", "content": "用户诉求"},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"action": "Policy_Search", "arguments": {"query": "申请条件"}},
                        ensure_ascii=False,
                    ),
                },
            ]
        }

        text = format_chatml_sample(sample)

        self.assertIn("<|im_start|>system", text)
        self.assertIn("<|im_start|>user", text)
        self.assertIn("<|im_start|>assistant", text)
        self.assertIn('"action": "Policy_Search"', text)
        self.assertTrue(text.endswith("<|im_end|>\n"))

    def test_load_sft_jsonl_reads_records_and_adds_text_field(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sft_samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "sft_00001",
                        "messages": [
                            {"role": "system", "content": "系统"},
                            {"role": "user", "content": "用户"},
                            {"role": "assistant", "content": '{"action": "Ask_User", "arguments": {}}'},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_sft_jsonl(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "sft_00001")
        self.assertIn("<|im_start|>assistant", records[0]["text"])

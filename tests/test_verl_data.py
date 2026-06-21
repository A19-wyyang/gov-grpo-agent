import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.verl_data import convert_grpo_jsonl_to_verl_parquet


class VerlDataTests(unittest.TestCase):
    def test_convert_grpo_jsonl_to_verl_parquet_writes_records_and_report(self):
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "grpo.jsonl"
            output_path = Path(temp_dir) / "train.parquet"
            report_path = Path(temp_dir) / "report.json"
            input_path.write_text(
                json.dumps(
                    {
                        "case_id": "case_1",
                        "prompt": "用户要办理公积金提取",
                        "responses": [
                            {"rollout_id": "r1", "trajectory": [{"action": "Policy_Search"}]},
                            {"rollout_id": "r2", "trajectory": [{"action": "Submit"}]},
                        ],
                        "rewards": [0.9, 0.2],
                        "advantages": [1.0, -1.0],
                        "reward_mean": 0.55,
                        "reward_std": 0.35,
                        "low_variance": False,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            report = convert_grpo_jsonl_to_verl_parquet(input_path, output_path, report_path)

            self.assertEqual(report["groups"], 1)
            self.assertEqual(report["responses"], 2)
            self.assertEqual(report["usable_groups"], 1)
            self.assertEqual(report["low_variance_groups"], 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(report["best_reward"], 0.9)

    def test_convert_grpo_jsonl_to_verl_parquet_rejects_all_low_variance_data(self):
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "grpo.jsonl"
            output_path = Path(temp_dir) / "train.parquet"
            input_path.write_text(
                json.dumps(
                    {
                        "case_id": "case_1",
                        "prompt": "用户要办理公积金提取",
                        "responses": [{"rollout_id": "r1", "trajectory": []}],
                        "rewards": [0.5],
                        "advantages": [0.0],
                        "reward_std": 0.0,
                        "low_variance": True,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No usable GRPO groups"):
                convert_grpo_jsonl_to_verl_parquet(input_path, output_path)

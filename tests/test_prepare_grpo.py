import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.prepare_grpo import prepare_grpo_dataset


class PrepareGrpoTests(unittest.TestCase):
    def test_prepare_grpo_dataset_writes_jsonl_and_report(self):
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "groups.json"
            output_path = Path(temp_dir) / "train.jsonl"
            report_path = Path(temp_dir) / "report.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_1",
                            "prompt": "用户诉求",
                            "reward_mean": 0.5,
                            "reward_std": 0.2,
                            "low_variance": False,
                            "responses": [
                                {
                                    "rollout_id": "r1",
                                    "trajectory": [{"action": "Policy_Search"}],
                                    "reward": 0.8,
                                    "advantage": 1.0,
                                },
                                {
                                    "rollout_id": "r2",
                                    "trajectory": [{"action": "Submit"}],
                                    "reward": 0.2,
                                    "advantage": -1.0,
                                },
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = prepare_grpo_dataset(input_path, output_path, report_path)

            self.assertEqual(report["groups"], 1)
            self.assertEqual(report["responses"], 2)
            self.assertEqual(report["usable_groups"], 1)
            self.assertEqual(report["low_variance_groups"], 0)
            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["case_id"], "case_1")
            self.assertEqual(record["prompt"], "用户诉求")
            self.assertEqual(record["rewards"], [0.8, 0.2])
            self.assertEqual(record["advantages"], [1.0, -1.0])
            self.assertTrue(report_path.exists())

    def test_prepare_grpo_dataset_marks_low_variance_groups_unusable(self):
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "groups.json"
            output_path = Path(temp_dir) / "train.jsonl"
            report_path = Path(temp_dir) / "report.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "case_2",
                            "prompt": "",
                            "reward_mean": 0.5,
                            "reward_std": 0.0,
                            "low_variance": True,
                            "responses": [
                                {"rollout_id": "r1", "trajectory": [], "reward": 0.5, "advantage": 0.0},
                                {"rollout_id": "r2", "trajectory": [], "reward": 0.5, "advantage": 0.0},
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = prepare_grpo_dataset(input_path, output_path, report_path)

            self.assertEqual(report["groups"], 1)
            self.assertEqual(report["usable_groups"], 0)
            self.assertEqual(report["low_variance_groups"], 1)

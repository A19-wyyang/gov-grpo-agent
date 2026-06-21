import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.train_grpo_verl import prepare_verl_training_job


class TrainGrpoVerlTests(unittest.TestCase):
    def test_prepare_verl_training_job_writes_data_config_script_and_manifest(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "grpo.jsonl"
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
                        "reward_std": 0.35,
                        "low_variance": False,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = prepare_verl_training_job(
                input_jsonl=input_path,
                work_dir=root / "job",
                model_path="Qwen/Qwen3-8B",
            )

            self.assertTrue(Path(manifest["train_parquet"]).exists())
            self.assertTrue(Path(manifest["config"]).exists())
            self.assertTrue(Path(manifest["run_script"]).exists())
            self.assertIn("verl.trainer.main_ppo", " ".join(manifest["command"]))
            self.assertEqual(manifest["data_report"]["usable_groups"], 1)

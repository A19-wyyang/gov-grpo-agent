import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gov_grpo_agent.train_grpo_verl import main, parse_gpu_ids, prepare_verl_training_job


class TrainGrpoVerlTests(unittest.TestCase):
    def test_parse_gpu_ids_normalizes_valid_list(self):
        self.assertEqual(parse_gpu_ids("4, 5,6,7"), [4, 5, 6, 7])

    def test_parse_gpu_ids_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_gpu_ids("4,5,4")

    def test_parse_gpu_ids_rejects_invalid_values(self):
        for value in ("", "4,,5", "4,x", "-1,4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_gpu_ids(value)

    @patch("gov_grpo_agent.train_grpo_verl.prepare_verl_training_job")
    @patch("builtins.print")
    def test_main_passes_selected_gpu_ids_to_job_preparation(self, _print, prepare_job):
        prepare_job.return_value = {"status": "prepared"}

        main(
            [
                "--input-jsonl",
                "train.jsonl",
                "--gpus",
                "4,5,6,7",
            ]
        )

        self.assertEqual(prepare_job.call_args.kwargs["gpu_ids"], [4, 5, 6, 7])

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
                gpu_ids=[4, 5, 6, 7],
            )

            self.assertTrue(Path(manifest["train_parquet"]).exists())
            self.assertTrue(Path(manifest["config"]).exists())
            self.assertTrue(Path(manifest["run_script"]).exists())
            self.assertIn("verl.trainer.main_ppo", " ".join(manifest["command"]))
            self.assertNotIn("--config-dir", manifest["command"])
            self.assertIn("algorithm.adv_estimator=grpo", manifest["command"])
            self.assertIn("trainer.n_gpus_per_node=4", manifest["command"])
            self.assertEqual(manifest["gpu_ids"], [4, 5, 6, 7])
            self.assertEqual(manifest["n_gpus_per_node"], 4)
            script = Path(manifest["run_script"]).read_text(encoding="utf-8")
            self.assertIn("export CUDA_VISIBLE_DEVICES=4,5,6,7", script)
            self.assertEqual(manifest["data_report"]["usable_groups"], 1)

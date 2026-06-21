import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.verl_config import build_verl_grpo_command, write_verl_grpo_config


class VerlConfigTests(unittest.TestCase):
    def test_write_verl_grpo_config_contains_grpo_and_custom_reward(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "verl_grpo.yaml"

            config = write_verl_grpo_config(
                output_path=config_path,
                model_path="Qwen/Qwen3-8B",
                train_files="artifacts/verl/train.parquet",
                val_files="artifacts/verl/train.parquet",
                save_dir="artifacts/verl_grpo",
                reward_path="gov_grpo_agent/verl_reward.py",
            )

            text = config_path.read_text(encoding="utf-8")
            self.assertEqual(config["algorithm"]["adv_estimator"], "grpo")
            self.assertIn("adv_estimator: grpo", text)
            self.assertIn("logger:", text)
            self.assertIn("- console", text)
            self.assertIn("custom_reward_function:", text)
            self.assertIn("path: gov_grpo_agent/verl_reward.py", text)
            self.assertIn("reward_model:", text)
            self.assertIn("num_workers: null", text)
            self.assertIn("reward_manager: null", text)
            self.assertIn("reward_loop_source: null", text)

    def test_build_verl_grpo_command_points_to_main_ppo_and_config(self):
        command = build_verl_grpo_command("configs/verl_grpo_qwen3_8b.yaml")

        self.assertEqual(command[0:3], ["python3", "-m", "verl.trainer.main_ppo"])
        self.assertIn("--config-dir", command)
        self.assertTrue(command[command.index("--config-dir") + 1].endswith("configs"))
        self.assertEqual(command[-2:], ["--config-name", "verl_grpo_qwen3_8b"])

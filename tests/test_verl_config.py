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
            self.assertEqual(config["custom_reward_function"], {"path": None, "name": None})
            self.assertIn("reward_model:", text)
            self.assertIn("num_workers: null", text)
            self.assertIn("reward_manager: null", text)
            self.assertIn("reward_loop_source: null", text)
            self.assertEqual(config["reward_model"]["reward_loop_module_path"], None)
            self.assertEqual(config["reward_model"]["reward_loop_class_name"], None)
            self.assertEqual(config["reward_model"]["enable"], None)
            self.assertEqual(config["reward_model"]["enable_resource_pool"], None)
            self.assertEqual(config["reward_model"]["n_gpus_per_node"], None)
            self.assertEqual(config["reward_model"]["nnodes"], None)
            self.assertEqual(config["reward_model"]["reward_kwargs"], None)
            self.assertEqual(config["reward_model"]["model"]["path"], None)
            self.assertEqual(config["reward_model"]["rollout"]["name"], None)
            self.assertEqual(config["sandbox_fusion"], {"url": None, "max_concurrent": None})

    def test_build_verl_grpo_command_uses_official_config_with_overrides(self):
        config = {
            "algorithm": {"adv_estimator": "grpo", "kl_ctrl": {"kl_coef": 0.03}},
            "reward": {
                "custom_reward_function": {
                    "path": "/repo/gov_grpo_agent/verl_reward.py",
                    "name": "compute_score",
                }
            },
            "data": {
                "train_files": "/job/data/train.parquet",
                "val_files": "/job/data/train.parquet",
                "max_prompt_length": 2048,
                "max_response_length": 512,
                "train_batch_size": 64,
            },
            "actor_rollout_ref": {
                "model": {"path": "Qwen/Qwen3-8B", "enable_gradient_checkpointing": True},
                "actor": {
                    "optim": {"lr": 1e-6},
                    "ppo_mini_batch_size": 16,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "use_kl_loss": True,
                    "kl_loss_coef": 0.03,
                },
                "rollout": {
                    "name": "vllm",
                    "n": 4,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "gpu_memory_utilization": 0.75,
                },
                "ref": {"log_prob_micro_batch_size_per_gpu": 1},
            },
            "trainer": {
                "project_name": "gov-grpo-agent",
                "experiment_name": "qwen3_8b_grpo",
                "logger": ["console", "tensorboard"],
                "default_local_dir": "/job/checkpoints",
                "total_epochs": 1,
                "save_freq": 10,
                "test_freq": 10,
                "nnodes": 1,
                "n_gpus_per_node": 8,
            },
        }

        command = build_verl_grpo_command(config)

        self.assertEqual(command[0:3], ["python3", "-m", "verl.trainer.main_ppo"])
        self.assertNotIn("--config-dir", command)
        self.assertNotIn("--config-name", command)
        self.assertIn("algorithm.adv_estimator=grpo", command)
        self.assertIn("data.train_files=/job/data/train.parquet", command)
        self.assertIn("reward.custom_reward_function.name=compute_score", command)
        self.assertIn("actor_rollout_ref.rollout.n=4", command)
        self.assertIn("trainer.logger=[console,tensorboard]", command)
        self.assertIn("trainer.n_gpus_per_node=8", command)

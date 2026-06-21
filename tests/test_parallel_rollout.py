import unittest
from pathlib import Path

from gov_grpo_agent.parallel_rollout import build_worker_commands, plan_case_shards


class ParallelRolloutTests(unittest.TestCase):
    def test_plan_case_shards_evenly_splits_cases_across_gpus(self):
        shards = plan_case_shards(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], total_cases=200)

        self.assertEqual(len(shards), 8)
        self.assertEqual(shards[0].gpu_id, 0)
        self.assertEqual(shards[0].case_offset, 0)
        self.assertEqual(shards[0].case_count, 25)
        self.assertEqual(shards[-1].gpu_id, 7)
        self.assertEqual(shards[-1].case_offset, 175)
        self.assertEqual(shards[-1].case_count, 25)

    def test_plan_case_shards_distributes_remainder_to_earlier_gpus(self):
        shards = plan_case_shards(gpu_ids=[0, 1, 2], total_cases=10)

        self.assertEqual(
            [(s.case_offset, s.case_count) for s in shards],
            [(0, 4), (4, 3), (7, 3)],
        )

    def test_build_worker_commands_include_gpu_binding_and_output_dir(self):
        commands = build_worker_commands(
            gpu_ids=[4, 5],
            total_cases=50,
            model_name_or_path="Qwen/Qwen3-8B",
            adapter_path="artifacts/qwen3_8b_sft_lora",
            output_root=Path("artifacts/parallel_rollout"),
            rollout_group_size=4,
            max_turns=8,
        )

        self.assertEqual(len(commands), 2)
        first = commands[0]
        self.assertEqual(first.env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(first.env["NCCL_P2P_DISABLE"], "1")
        self.assertIn("--case-offset", first.args)
        self.assertIn("0", first.args)
        self.assertIn("--case-count", first.args)
        self.assertIn("25", first.args)
        self.assertIn("artifacts/parallel_rollout/gpu4", first.args)

    def test_build_worker_commands_include_sampling_parameters(self):
        commands = build_worker_commands(
            gpu_ids=[0],
            total_cases=4,
            model_name_or_path="Qwen/Qwen3-8B",
            adapter_path="adapter",
            output_root=Path("rollouts"),
            rollout_group_size=4,
            max_turns=8,
            temperature=1.1,
            top_p=0.92,
        )

        args = commands[0].args
        self.assertIn("--do-sample", args)
        self.assertEqual(args[args.index("--temperature") + 1], "1.1")
        self.assertEqual(args[args.index("--top-p") + 1], "0.92")

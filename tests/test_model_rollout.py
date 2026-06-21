import json
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.model_rollout import run_model_rollout


class CyclingGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        index = (self.calls - 1) % 4
        if index == 0:
            return {
                "action": "Policy_Search",
                "arguments": {
                    "service_item": "住房公积金提取",
                    "city": "杭州",
                    "query": "租房提取条件 材料",
                },
            }
        if index == 1:
            return {"action": "Eligibility_Check", "arguments": {}}
        if index == 2:
            return {"action": "Material_Check", "arguments": {}}
        return {"action": "Submit", "arguments": {"final_answer": "placeholder"}}


class BrokenGenerator:
    def generate(self, prompt):
        raise ValueError("no JSON object found")


class ModelRolloutTests(unittest.TestCase):
    def test_run_model_rollout_writes_model_trajectory_outputs(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "model_rollout"

            summary = run_model_rollout(
                action_generator=CyclingGenerator(),
                output_dir=output_dir,
                case_count=3,
                rollout_group_size=1,
            )

            self.assertEqual(summary["cases"], 3)
            self.assertEqual(summary["trajectories"], 3)
            self.assertEqual(summary["grpo_groups"], 3)
            self.assertTrue((output_dir / "model_trajectories.jsonl").exists())
            self.assertTrue((output_dir / "model_reward_reports.jsonl").exists())
            self.assertTrue((output_dir / "model_grpo_groups.json").exists())
            self.assertTrue((output_dir / "model_metrics.json").exists())

            first = json.loads(
                (output_dir / "model_trajectories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(first["steps"][0]["arguments"]["service_item"], "租房提取公积金")

    def test_run_model_rollout_supports_case_offset_and_progress_logging(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "model_rollout"
            progress = StringIO()

            summary = run_model_rollout(
                action_generator=CyclingGenerator(),
                output_dir=output_dir,
                case_count=2,
                rollout_group_size=1,
                case_offset=3,
                progress_stream=progress,
            )

            self.assertEqual(summary["case_offset"], 3)
            self.assertEqual(summary["cases"], 2)
            self.assertIn("[rollout] 1/2", progress.getvalue())
            self.assertIn("[rollout] 2/2", progress.getvalue())

            lines = (output_dir / "model_trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["case_id"], "talent_subsidy_0004")

    def test_run_model_rollout_records_invalid_model_generation_without_crashing(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "model_rollout"

            summary = run_model_rollout(
                action_generator=BrokenGenerator(),
                output_dir=output_dir,
                case_count=1,
                rollout_group_size=1,
            )

            self.assertEqual(summary["trajectories"], 1)
            self.assertEqual(summary["metrics"]["invalid_action_rate"], 1.0)
            trajectory = json.loads(
                (output_dir / "model_trajectories.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(trajectory["steps"][0]["action"], "")
            self.assertIn("no JSON object found", trajectory["steps"][0]["observation"]["error"])

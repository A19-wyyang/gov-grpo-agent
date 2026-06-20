import json
import unittest

from gov_grpo_agent.cli import run_mvp


class CliTests(unittest.TestCase):
    def test_run_mvp_writes_cases_rollouts_rewards_and_groups(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "mvp"

            summary = run_mvp(output_dir=output_dir, case_count=20, rollout_group_size=4)

            self.assertEqual(summary["cases"], 20)
            self.assertEqual(summary["trajectories"], 80)
            self.assertEqual(summary["grpo_groups"], 20)
            self.assertTrue((output_dir / "cases.jsonl").exists())
            self.assertTrue((output_dir / "trajectories.jsonl").exists())
            self.assertTrue((output_dir / "reward_reports.jsonl").exists())
            self.assertTrue((output_dir / "grpo_groups.json").exists())

            groups = json.loads((output_dir / "grpo_groups.json").read_text(encoding="utf-8"))
            self.assertEqual(len(groups), 20)
            self.assertTrue(all(len(group["responses"]) == 4 for group in groups))

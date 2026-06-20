import unittest

from gov_grpo_agent.grpo import build_grpo_groups


class GrpoTests(unittest.TestCase):
    def test_build_grpo_groups_computes_group_relative_advantages(self):
        trajectories = [
            {"case_id": "case_1", "rollout_id": "r1", "steps": []},
            {"case_id": "case_1", "rollout_id": "r2", "steps": []},
            {"case_id": "case_1", "rollout_id": "r3", "steps": []},
            {"case_id": "case_1", "rollout_id": "r4", "steps": []},
        ]
        rewards = {
            "r1": {"reward": 0.2},
            "r2": {"reward": 0.4},
            "r3": {"reward": 0.8},
            "r4": {"reward": 1.0},
        }

        groups = build_grpo_groups(trajectories, rewards)

        self.assertEqual(len(groups), 1)
        advantages = [response["advantage"] for response in groups[0]["responses"]]
        self.assertEqual(round(sum(advantages), 7), 0)
        self.assertLess(advantages[0], 0)
        self.assertGreater(advantages[-1], 0)
        self.assertGreater(groups[0]["reward_std"], 0)

    def test_build_grpo_groups_handles_zero_variance_without_nan(self):
        trajectories = [
            {"case_id": "case_2", "rollout_id": "r1", "steps": []},
            {"case_id": "case_2", "rollout_id": "r2", "steps": []},
        ]
        rewards = {"r1": {"reward": 0.5}, "r2": {"reward": 0.5}}

        groups = build_grpo_groups(trajectories, rewards)

        self.assertEqual(
            [response["advantage"] for response in groups[0]["responses"]],
            [0.0, 0.0],
        )
        self.assertTrue(groups[0]["low_variance"])

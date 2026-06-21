import unittest

from gov_grpo_agent.verl_reward import compute_score


class VerlRewardTests(unittest.TestCase):
    def test_compute_score_rewards_valid_action_matching_expected_sequence(self):
        score = compute_score(
            solution_str='{"action": "Policy_Search", "arguments": {"service_item": "租房提取公积金"}}',
            ground_truth={"expected_actions": ["Policy_Search", "Eligibility_Check"], "best_reward": 0.9},
        )

        self.assertGreaterEqual(score, 0.7)

    def test_compute_score_penalizes_invalid_json(self):
        score = compute_score(
            solution_str="我觉得可以直接提交",
            ground_truth={"expected_actions": ["Policy_Search"], "best_reward": 0.9},
        )

        self.assertEqual(score, 0.0)

import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.rewards import score_trajectory
from gov_grpo_agent.runtime import AgentRuntime, RuleBasedPolicy


class RewardTests(unittest.TestCase):
    def test_reward_scores_complete_tool_grounded_trajectory_highly(self):
        case = build_case("housing_fund", 20, "material_missing")
        trajectory = AgentRuntime(policy=RuleBasedPolicy()).run_case(case)

        report = score_trajectory(case, trajectory)

        self.assertGreaterEqual(report["verifier_score"], 0.9)
        self.assertGreaterEqual(report["judge_score"], 0.8)
        self.assertEqual(report["penalty"], 0)
        self.assertGreaterEqual(report["reward"], 0.85)
        self.assertEqual(report["failure_reasons"], [])

    def test_reward_penalizes_missing_tool_before_submit(self):
        case = build_case("housing_fund", 21, "material_missing")
        trajectory = {
            "case_id": case["case_id"],
            "rollout_id": "manual_bad",
            "steps": [
                {
                    "turn": 1,
                    "action": "Policy_Search",
                    "arguments": {},
                    "observation": {},
                },
                {
                    "turn": 2,
                    "action": "Submit",
                    "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
                    "observation": {},
                },
            ],
            "final_answer": case["hidden_truth"]["final_decision"],
            "metadata": {"path_type": case["path_type"], "difficulty": case["difficulty"]},
        }

        report = score_trajectory(case, trajectory)

        self.assertGreaterEqual(report["penalty"], 0.25)
        self.assertIn("missing_required_tool:Eligibility_Check", report["failure_reasons"])
        self.assertLess(report["reward"], 0.75)

    def test_reward_penalizes_premature_submit_with_missing_slots(self):
        case = build_case("housing_fund", 22, "missing_information")
        trajectory = {
            "case_id": case["case_id"],
            "rollout_id": "manual_premature",
            "steps": [
                {
                    "turn": 1,
                    "action": "Submit",
                    "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
                    "observation": {},
                }
            ],
            "final_answer": case["hidden_truth"]["final_decision"],
            "metadata": {"path_type": case["path_type"], "difficulty": case["difficulty"]},
        }

        report = score_trajectory(case, trajectory)

        self.assertGreaterEqual(report["penalty"], 0.3)
        self.assertIn("premature_submit:missing_slots", report["failure_reasons"])

    def test_reward_handles_non_string_final_answer_without_crashing(self):
        case = build_case("housing_fund", 23, "simple_success")
        trajectory = {
            "case_id": case["case_id"],
            "rollout_id": "manual_bool_answer",
            "steps": [
                {
                    "turn": 1,
                    "action": "Policy_Search",
                    "arguments": {},
                    "observation": {},
                },
                {
                    "turn": 2,
                    "action": "Eligibility_Check",
                    "arguments": {},
                    "observation": {},
                },
                {
                    "turn": 3,
                    "action": "Material_Check",
                    "arguments": {},
                    "observation": {},
                },
                {
                    "turn": 4,
                    "action": "Submit",
                    "arguments": {"final_answer": True},
                    "observation": {"final_answer": True},
                },
            ],
            "final_answer": True,
            "metadata": {"path_type": case["path_type"], "difficulty": case["difficulty"]},
        }

        report = score_trajectory(case, trajectory)

        self.assertEqual(report["judge_score"], 0.0)
        self.assertIn("final_answer:not_string", report["failure_reasons"])

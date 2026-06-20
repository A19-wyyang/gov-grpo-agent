import unittest

from gov_grpo_agent.data import build_mvp_cases
from gov_grpo_agent.evaluation import compute_metrics
from gov_grpo_agent.rewards import score_trajectory
from gov_grpo_agent.runtime import AgentRuntime, RuleBasedPolicy


class EvaluationTests(unittest.TestCase):
    def test_compute_metrics_reports_core_mvp_indicators(self):
        cases = build_mvp_cases(limit=10)
        runtime = AgentRuntime(policy=RuleBasedPolicy())
        trajectories = [runtime.run_case(case) for case in cases]
        reports = [score_trajectory(case, trajectory) for case, trajectory in zip(cases, trajectories)]

        metrics = compute_metrics(cases, trajectories, reports)

        self.assertIn("success_at_1", metrics)
        self.assertIn("required_tool_recall", metrics)
        self.assertIn("premature_submit_rate", metrics)
        self.assertIn("missing_tool_rate", metrics)
        self.assertIn("material_check_call_rate", metrics)
        self.assertIn("final_decision_accuracy", metrics)
        self.assertEqual(metrics["required_tool_recall"], 1.0)
        self.assertEqual(metrics["missing_tool_rate"], 0.0)

    def test_compute_metrics_detects_missing_tool_failures(self):
        case = build_mvp_cases(limit=1)[0]
        trajectory = {
            "case_id": case["case_id"],
            "rollout_id": "bad",
            "steps": [
                {
                    "turn": 1,
                    "action": "Submit",
                    "arguments": {"final_answer": "直接提交"},
                    "observation": {},
                }
            ],
            "final_answer": "直接提交",
            "metadata": {"path_type": case["path_type"], "difficulty": case["difficulty"]},
        }
        report = score_trajectory(case, trajectory)

        metrics = compute_metrics([case], [trajectory], [report])

        self.assertEqual(metrics["required_tool_recall"], 0.0)
        self.assertEqual(metrics["missing_tool_rate"], 1.0)

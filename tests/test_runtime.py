import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.runtime import AgentRuntime, RuleBasedPolicy
from gov_grpo_agent.schemas import ActionValidationError, validate_trajectory


class RuntimeTests(unittest.TestCase):
    def test_runtime_runs_complete_material_missing_trajectory(self):
        case = build_case("housing_fund", 10, "material_missing")
        trajectory = AgentRuntime(policy=RuleBasedPolicy(), max_turns=8).run_case(case)

        actions = [step["action"] for step in trajectory["steps"]]
        self.assertEqual(
            actions,
            ["Policy_Search", "Eligibility_Check", "Material_Check", "Submit"],
        )
        self.assertEqual(trajectory["final_answer"], case["hidden_truth"]["final_decision"])
        self.assertTrue(validate_trajectory(trajectory))

    def test_runtime_asks_user_before_tooling_when_required_slots_are_missing(self):
        case = build_case("housing_fund", 11, "missing_information")
        trajectory = AgentRuntime(policy=RuleBasedPolicy(), max_turns=8).run_case(case)

        self.assertEqual(trajectory["steps"][0]["action"], "Ask_User")
        self.assertIn("city", trajectory["steps"][0]["arguments"]["slots"])

    def test_action_validator_rejects_unknown_action(self):
        with self.assertRaises(ActionValidationError):
            AgentRuntime(policy=RuleBasedPolicy()).execute_action(
                build_case("housing_fund", 12, "simple_success"),
                {"action": "Answer_Directly", "arguments": {}},
            )

    def test_runtime_handles_terminal_action_missing_final_answer(self):
        class MissingFinalAnswerPolicy:
            def next_action(self, case, steps):
                return {"action": "Submit", "arguments": {}}

        trajectory = AgentRuntime(policy=MissingFinalAnswerPolicy(), max_turns=1).run_case(
            build_case("housing_fund", 13, "simple_success")
        )

        self.assertEqual(trajectory["final_answer"], "")
        self.assertEqual(trajectory["steps"][0]["arguments"]["final_answer"], "")

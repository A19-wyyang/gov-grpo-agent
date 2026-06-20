import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.model_policy import ModelActionPolicy
from gov_grpo_agent.runtime import AgentRuntime


class FakeActionGenerator:
    def __init__(self, actions):
        self.actions = list(actions)
        self.queries = []

    def generate(self, prompt):
        self.queries.append(prompt)
        return self.actions.pop(0)


class ModelPolicyTests(unittest.TestCase):
    def test_model_policy_normalizes_action_arguments_before_runtime_tool_call(self):
        case = build_case("housing_fund", 1, "material_missing")
        generator = FakeActionGenerator(
            [
                {
                    "action": "Policy_Search",
                    "arguments": {
                        "service_item": "住房公积金提取",
                        "city": "杭州",
                        "query": "租房提取条件 材料",
                    },
                },
                {"action": "Eligibility_Check", "arguments": {}},
                {"action": "Material_Check", "arguments": {}},
                {
                    "action": "Submit",
                    "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
                },
            ]
        )

        trajectory = AgentRuntime(policy=ModelActionPolicy(generator), max_turns=8).run_case(case)

        self.assertEqual(
            trajectory["steps"][0]["arguments"]["service_item"],
            "租房提取公积金",
        )
        self.assertIn("conditions", trajectory["steps"][0]["observation"])
        self.assertEqual(trajectory["final_answer"], case["hidden_truth"]["final_decision"])

    def test_model_policy_includes_case_context_in_prompt(self):
        case = build_case("housing_fund", 2, "simple_success")
        generator = FakeActionGenerator(
            [
                {
                    "action": "Submit",
                    "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
                }
            ]
        )

        AgentRuntime(policy=ModelActionPolicy(generator), max_turns=1).run_case(case)

        self.assertIn(case["service_item"], generator.queries[0])
        self.assertIn(case["user_initial_query"], generator.queries[0])

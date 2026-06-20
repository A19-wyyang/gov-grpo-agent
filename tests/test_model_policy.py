import unittest

from gov_grpo_agent.data import build_case
from gov_grpo_agent.model_policy import ModelActionPolicy, build_policy_prompt
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

    def test_policy_prompt_lists_required_and_remaining_tools(self):
        case = build_case("housing_fund", 3, "simple_success")
        steps = [{"action": "Policy_Search", "observation": {}}]

        prompt = build_policy_prompt(case, steps)

        self.assertIn("必要工具", prompt)
        self.assertIn("剩余必要工具", prompt)
        self.assertIn("Eligibility_Check", prompt)
        self.assertIn("Material_Check", prompt)
        self.assertIn("禁止 Submit/Refuse", prompt)

    def test_model_policy_can_enforce_remaining_required_tools_before_submit(self):
        case = build_case("housing_fund", 4, "simple_success")
        generator = FakeActionGenerator(
            [
                {
                    "action": "Submit",
                    "arguments": {"final_answer": "过早提交"},
                }
            ]
        )
        steps = [{"action": "Policy_Search", "observation": {}}]

        action = ModelActionPolicy(generator, enforce_required_tools=True).next_action(case, steps)

        self.assertEqual(action["action"], "Eligibility_Check")
        self.assertEqual(action["arguments"], {})

from gov_grpo_agent.schemas import validate_action
from gov_grpo_agent.tools import TOOL_REGISTRY


class RuleBasedPolicy:
    """Deterministic baseline policy for MVP rollout generation."""

    def next_action(self, case, steps):
        called = [step["action"] for step in steps]
        missing_slots = case["hidden_truth"]["missing_slots"]
        if missing_slots and "Ask_User" not in called:
            return {"action": "Ask_User", "arguments": {"slots": list(missing_slots)}}
        for tool_name in case["hidden_truth"]["required_tools"]:
            if tool_name not in called:
                return {
                    "action": tool_name,
                    "arguments": self._tool_arguments(tool_name, case),
                }
        final_action = "Submit" if case["hidden_truth"]["eligible"] else "Refuse"
        return {
            "action": final_action,
            "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
        }

    def _tool_arguments(self, tool_name, case):
        if tool_name == "Policy_Search":
            return {
                "service_item": case["service_item"],
                "city": case["user_profile"].get("city"),
                "query": "申请条件 材料",
            }
        return {}


class AgentRuntime:
    def __init__(self, policy, max_turns=8):
        self.policy = policy
        self.max_turns = max_turns

    def run_case(self, case, rollout_id=None):
        steps = []
        final_answer = ""
        for turn in range(1, self.max_turns + 1):
            action = self.policy.next_action(case, steps)
            observation = self.execute_action(case, action)
            steps.append(
                {
                    "turn": turn,
                    "action": action["action"],
                    "arguments": action["arguments"],
                    "observation": observation,
                }
            )
            if action["action"] in {"Submit", "Refuse"}:
                final_answer = action["arguments"].setdefault("final_answer", "")
                steps[-1]["arguments"]["final_answer"] = final_answer
                if isinstance(steps[-1]["observation"], dict):
                    steps[-1]["observation"].setdefault("final_answer", final_answer)
                break
        else:
            final_answer = "超过最大轮次，办理轨迹失败。"

        return {
            "case_id": case["case_id"],
            "rollout_id": rollout_id or f"{case['case_id']}_r01",
            "steps": steps,
            "final_answer": final_answer,
            "metadata": {
                "path_type": case["path_type"],
                "difficulty": case["difficulty"],
                "error_type": case["error_type"],
            },
        }

    def execute_action(self, case, action):
        validate_action(action)
        action_name = action["action"]
        if action_name == "Ask_User":
            slots = action["arguments"].get("slots", [])
            return {
                "filled_slots": {
                    slot: case["user_profile"].get(slot) or "杭州" for slot in slots
                }
            }
        if action_name in TOOL_REGISTRY:
            return TOOL_REGISTRY[action_name].run(action["arguments"], case)
        if action_name in {"Submit", "Refuse"}:
            return {"finalized": True, "final_answer": action["arguments"].get("final_answer", "")}
        if action_name == "Risk_Check":
            return {"risk_level": "low", "risks": []}
        return {}

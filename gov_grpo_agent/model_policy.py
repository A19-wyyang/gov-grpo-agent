from gov_grpo_agent.normalization import normalize_action_arguments
from gov_grpo_agent.schemas import validate_action


class ModelActionPolicy:
    def __init__(self, action_generator, enforce_required_tools=False):
        self.action_generator = action_generator
        self.enforce_required_tools = enforce_required_tools

    def next_action(self, case, steps):
        prompt = build_policy_prompt(case, steps)
        action = self.action_generator.generate(prompt)
        action = normalize_action_arguments(action)
        if self.enforce_required_tools:
            action = enforce_required_tool_order(case, steps, action)
        validate_action(action)
        return action


def build_policy_prompt(case, steps):
    called_actions = [step["action"] for step in steps]
    required_tools = case["hidden_truth"]["required_tools"]
    remaining_tools = [tool for tool in required_tools if tool not in called_actions]
    observations = [
        {"action": step["action"], "observation": step.get("observation", {})}
        for step in steps
    ]
    return (
        "你是政务办理Agent。请根据当前case和已执行轨迹输出下一步合法JSON动作。\n"
        f"事项：{case['service_item']}\n"
        f"用户诉求：{case['user_initial_query']}\n"
        f"必要工具：{required_tools}\n"
        f"剩余必要工具：{remaining_tools}\n"
        f"已调用动作：{called_actions}\n"
        f"工具观察：{observations}\n"
        "动作空间：Ask_User, Policy_Search, Eligibility_Check, Material_Check, Risk_Check, Submit, Refuse。\n"
        "禁止 Submit/Refuse：只要剩余必要工具非空，下一步必须调用剩余必要工具中的第一个。\n"
        "只输出JSON，不要输出解释。"
    )


def enforce_required_tool_order(case, steps, action):
    called_actions = [step["action"] for step in steps]
    missing_slots = case["hidden_truth"]["missing_slots"]
    if missing_slots and "Ask_User" not in called_actions:
        return {"action": "Ask_User", "arguments": {"slots": list(missing_slots)}}
    if action["action"] not in {"Submit", "Refuse"}:
        return action
    for tool_name in case["hidden_truth"]["required_tools"]:
        if tool_name not in called_actions:
            if tool_name == "Policy_Search":
                return {
                    "action": "Policy_Search",
                    "arguments": {
                        "service_item": case["service_item"],
                        "city": case["user_profile"].get("city"),
                        "query": "申请条件 材料",
                    },
                }
            return {"action": tool_name, "arguments": {}}
    return action

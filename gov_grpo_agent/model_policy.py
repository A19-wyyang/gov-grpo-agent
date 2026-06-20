from gov_grpo_agent.normalization import normalize_action_arguments
from gov_grpo_agent.schemas import validate_action


class ModelActionPolicy:
    def __init__(self, action_generator):
        self.action_generator = action_generator

    def next_action(self, case, steps):
        prompt = build_policy_prompt(case, steps)
        action = self.action_generator.generate(prompt)
        action = normalize_action_arguments(action)
        validate_action(action)
        return action


def build_policy_prompt(case, steps):
    called_actions = [step["action"] for step in steps]
    observations = [
        {"action": step["action"], "observation": step.get("observation", {})}
        for step in steps
    ]
    return (
        "你是政务办理Agent。请根据当前case和已执行轨迹输出下一步合法JSON动作。\n"
        f"事项：{case['service_item']}\n"
        f"用户诉求：{case['user_initial_query']}\n"
        f"已调用动作：{called_actions}\n"
        f"工具观察：{observations}\n"
        "动作空间：Ask_User, Policy_Search, Eligibility_Check, Material_Check, Risk_Check, Submit, Refuse。\n"
        "只输出JSON，不要输出解释。"
    )

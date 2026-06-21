import json

from gov_grpo_agent.infer_sft import parse_action_from_generation


def compute_score(solution_str, ground_truth=None, **kwargs):
    ground_truth = ground_truth or {}
    try:
        action = parse_action_from_generation(solution_str)
    except (ValueError, json.JSONDecodeError, TypeError):
        return 0.0

    expected_actions = ground_truth.get("expected_actions", [])
    score = 0.35
    if action["action"] in expected_actions:
        score += 0.35
        if expected_actions and action["action"] == expected_actions[0]:
            score += 0.15
    if action["action"] in {"Submit", "Refuse"} and "final_answer" in action.get("arguments", {}):
        score += 0.15
    if action["arguments"]:
        score += 0.05
    best_reward = float(ground_truth.get("best_reward", 1.0) or 1.0)
    return round(min(1.0, score * max(best_reward, 0.1)), 4)

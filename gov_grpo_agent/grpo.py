from collections import defaultdict
from math import sqrt


def build_grpo_groups(trajectories, reward_reports):
    grouped = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory["case_id"]].append(trajectory)

    groups = []
    for case_id, case_trajectories in grouped.items():
        reward_values = [
            float(reward_reports[trajectory["rollout_id"]]["reward"])
            for trajectory in case_trajectories
        ]
        mean = sum(reward_values) / len(reward_values)
        variance = sum((value - mean) ** 2 for value in reward_values) / len(reward_values)
        std = sqrt(variance)
        low_variance = std < 1e-8
        responses = []
        for trajectory, reward in zip(case_trajectories, reward_values):
            advantage = 0.0 if low_variance else (reward - mean) / std
            responses.append(
                {
                    "rollout_id": trajectory["rollout_id"],
                    "trajectory": trajectory["steps"],
                    "reward": reward,
                    "advantage": round(advantage, 7),
                }
            )
        groups.append(
            {
                "case_id": case_id,
                "prompt": case_trajectories[0].get("prompt", ""),
                "reward_mean": round(mean, 7),
                "reward_std": round(std, 7),
                "low_variance": low_variance,
                "responses": responses,
            }
        )
    return groups

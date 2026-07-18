from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pvariance
from typing import Any

from .grpo import build_grpo_groups
from .io_utils import load_cases, write_jsonl
from .policies import POLICIES
from .rollout import rollout_case, rollout_cases
from .scoring import reward_profile, score_trajectory


TRAINING_POLICIES = [
    "careful_policy",
    "risky_policy",
    "conservative_refuse_policy",
    "judge_hacker_policy",
]


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    reward_profile_name: str
    entropy_bonus: float
    epochs: int = 24
    rollouts_per_case: int = 12
    learning_rate: float = 0.55
    seed: int = 7


EXPERIMENTS = {
    "before_fix": TrainingConfig(
        name="before_fix",
        reward_profile_name="collapse_prone",
        entropy_bonus=0.0,
    ),
    "after_fix": TrainingConfig(
        name="after_fix",
        reward_profile_name="hardened",
        # This coefficient is on policy-mode logits in the simulator, not token loss.
        entropy_bonus=2.5,
    ),
}


def train_policy_mixture(cases_dir: Path, config: TrainingConfig) -> list[dict[str, Any]]:
    """Simulate the policy-level effect of GRPO without requiring a GPU model."""
    cases = load_cases(cases_dir)
    reward_config = reward_profile(config.reward_profile_name)
    rng = random.Random(config.seed)
    logits = {
        "careful_policy": 0.0,
        "risky_policy": -0.10,
        "conservative_refuse_policy": 0.35,
        "judge_hacker_policy": 0.05,
    }
    history = []
    run_index = 0

    for epoch in range(1, config.epochs + 1):
        probabilities = _softmax(logits)
        trajectories = []
        for case in cases:
            for _ in range(config.rollouts_per_case):
                run_index += 1
                policy_name = _sample_policy(rng, probabilities)
                trajectories.append(
                    rollout_case(
                        case=case,
                        policy_name=policy_name,
                        policy=POLICIES[policy_name],
                        run_index=run_index,
                    )
                )

        case_map = {case.case_id: case for case in cases}
        scores = [
            score_trajectory(case_map[item.case_id], item.to_dict(), reward_config)
            for item in trajectories
        ]
        groups = build_grpo_groups(scores)
        advantages = {
            row["trajectory_id"]: row["advantage"]
            for group in groups
            for row in group["trajectories"]
        }
        advantage_by_policy: dict[str, list[float]] = defaultdict(list)
        for score in scores:
            advantage_by_policy[score["policy_name"]].append(
                advantages[score["trajectory_id"]]
            )

        entropy = _entropy(probabilities)
        for policy_name in TRAINING_POLICIES:
            policy_advantages = advantage_by_policy.get(policy_name, [])
            if policy_advantages:
                logits[policy_name] += config.learning_rate * mean(policy_advantages)
            entropy_gradient = -probabilities[policy_name] * (
                math.log(max(probabilities[policy_name], 1e-12)) + entropy
            )
            logits[policy_name] += config.entropy_bonus * entropy_gradient

        final_action_counts = Counter(
            item.final_decision["type"] for item in trajectories
        )
        required_tool_rate = mean(
            int(not score["missing_tools"]) for score in scores
        )
        history.append(
            {
                "experiment": config.name,
                "epoch": epoch,
                "reward_profile": config.reward_profile_name,
                "entropy_bonus": config.entropy_bonus,
                "mean_reward": round(mean(score["reward"] for score in scores), 4),
                "reward_variance": round(pvariance(score["reward"] for score in scores), 4),
                "mean_group_reward_std": round(mean(group["reward_std"] for group in groups), 4),
                "required_tool_rate": round(required_tool_rate, 4),
                "policy_entropy": round(entropy, 4),
                "policy_distribution": {
                    name: round(probabilities[name], 4) for name in TRAINING_POLICIES
                },
                "final_action_distribution": {
                    name: round(count / len(trajectories), 4)
                    for name, count in sorted(final_action_counts.items())
                },
            }
        )
    return history


def run_training_comparison(cases_dir: Path, out_dir: Path) -> Path:
    histories = {
        name: train_policy_mixture(cases_dir, config)
        for name, config in EXPERIMENTS.items()
    }
    for name, history in histories.items():
        write_jsonl(out_dir / "experiments" / f"{name}.jsonl", history)

    cases = load_cases(cases_dir)
    hacking_trajectories = rollout_cases(cases, policy_names=["judge_hacker_policy"])
    hacking_rows = []
    for trajectory in hacking_trajectories:
        case = next(case for case in cases if case.case_id == trajectory.case_id)
        trajectory_dict = trajectory.to_dict()
        hacking_rows.append(
            {
                "case_id": case.case_id,
                "trajectory_id": trajectory.trajectory_id,
                "collapse_prone": score_trajectory(
                    case, trajectory_dict, reward_profile("collapse_prone")
                ),
                "hardened": score_trajectory(
                    case, trajectory_dict, reward_profile("hardened")
                ),
            }
        )
    write_jsonl(out_dir / "experiments" / "reward_hacking_comparison.jsonl", hacking_rows)
    report_path = out_dir / "experiments" / "report.md"
    _write_experiment_report(report_path, histories, hacking_rows)
    return report_path


def _write_experiment_report(
    output_path: Path,
    histories: dict[str, list[dict[str, Any]]],
    hacking_rows: list[dict[str, Any]],
) -> None:
    before = histories["before_fix"][-1]
    after = histories["after_fix"][-1]
    lines = [
        "# GRPO 故障复现实验",
        "",
        "此处是策略混合分布模拟器，用于低成本解释训练动力学；它不是对 8B 模型做反向传播。",
        "",
        "## 策略坍塌对照",
        "",
        "| 实验 | Reward profile | Entropy bonus | 最终 entropy | 必要工具调用率 | Refuse 占比 | Careful policy 占比 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        _experiment_row(before),
        _experiment_row(after),
        "",
        "`before_fix` 故意使用不合理的延迟惩罚、过高 Judge 权重和较弱拒答惩罚，短路径拒答更容易获得高分。",
        "`after_fix` 强制必要工具调用、降低 Judge 权重，并加入 entropy bonus 保留探索。",
        "",
        "## Reward hacking 对照",
        "",
        "`judge_hacker_policy` 会给出流畅建议，但跳过 `MATERIAL_CHECK` 和 `RISK_CHECK`。",
        "",
        "| Case | Vulnerable reward | Hardened reward | Hardened missing tools |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in hacking_rows:
        missing = ", ".join(row["hardened"]["missing_tools"]) or "无"
        lines.append(
            f"| {row['case_id']} | {row['collapse_prone']['reward']} | "
            f"{row['hardened']['reward']} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## 生产训练替换点",
            "",
            "本地模拟器把一个 rollout 抽象为行为策略名。生产环境中应替换为模型按当前参数采样的 token 序列，",
            "保留相同的 case 分组、trajectory log、Verifier/Judge 聚合、组内标准化 advantage 和监控指标。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _experiment_row(row: dict[str, Any]) -> str:
    distribution = row["policy_distribution"]
    refuse_rate = row["final_action_distribution"].get("REFUSE", 0.0)
    return (
        f"| {row['experiment']} | {row['reward_profile']} | {row['entropy_bonus']} | "
        f"{row['policy_entropy']} | {row['required_tool_rate']} | {refuse_rate} | "
        f"{distribution['careful_policy']} |"
    )


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    maximum = max(logits.values())
    exponentials = {name: math.exp(value - maximum) for name, value in logits.items()}
    total = sum(exponentials.values())
    return {name: value / total for name, value in exponentials.items()}


def _sample_policy(rng: random.Random, probabilities: dict[str, float]) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for name in TRAINING_POLICIES:
        cumulative += probabilities[name]
        if threshold <= cumulative:
            return name
    return TRAINING_POLICIES[-1]


def _entropy(probabilities: dict[str, float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities.values())

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import load_cases, read_jsonl, write_jsonl
from .models import GovCase


@dataclass(frozen=True)
class RewardConfig:
    name: str
    verifier_weight: float
    judge_weight: float
    process_weight: float
    penalty_weights: dict[str, float]
    enforce_required_tools: bool = True
    step_cost: float = 0.0


REWARD_PROFILES = {
    "hardened": RewardConfig(
        name="hardened",
        verifier_weight=0.70,
        judge_weight=0.20,
        process_weight=0.10,
        penalty_weights={
            "early_submit": 0.25,
            "missing_required_tool": 0.35,
            "repeated_question": 0.10,
            "over_refuse": 0.25,
            "wrong_final_action": 0.35,
            "missing_slots_at_final": 0.20,
            "max_steps_exceeded": 0.20,
        },
    ),
    "collapse_prone": RewardConfig(
        name="collapse_prone",
        verifier_weight=0.45,
        judge_weight=0.45,
        process_weight=0.10,
        penalty_weights={
            "early_submit": 0.10,
            "missing_required_tool": 0.00,
            "repeated_question": 0.05,
            "over_refuse": 0.02,
            "wrong_final_action": 0.05,
            "missing_slots_at_final": 0.00,
            "max_steps_exceeded": 0.10,
        },
        enforce_required_tools=False,
        step_cost=0.05,
    ),
}


def reward_profile(name: str) -> RewardConfig:
    try:
        return REWARD_PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(REWARD_PROFILES))
        raise ValueError(f"unknown reward profile {name!r}; choose from: {supported}") from exc


def verify_trajectory(
    case: GovCase,
    trajectory: dict[str, Any],
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    config = config or reward_profile("hardened")
    final_action = trajectory["final_decision"]["type"]
    expected_action = case.expected_result["final_action"]
    final_step = trajectory["steps"][-1] if trajectory["steps"] else {}
    final_missing = final_step.get("slot_status", {}).get("missing", case.required_slots)
    tool_history = _tool_history(trajectory)
    failure_counter = Counter(trajectory.get("failure_tags", []))
    missing_tools = [tool for tool in case.required_tools if tool not in tool_history]

    penalties = {
        "early_submit": int(failure_counter["early_submit"] > 0),
        "missing_required_tool": int(bool(missing_tools)),
        "repeated_question": int(failure_counter["repeated_question"] > 0),
        "over_refuse": int(final_action == "REFUSE" and expected_action == "SUBMIT"),
        "wrong_final_action": int(final_action != expected_action),
        "missing_slots_at_final": int(bool(final_missing)),
        "max_steps_exceeded": int(failure_counter["max_steps_exceeded"] > 0),
    }

    checks = {
        "final_action_correct": final_action == expected_action,
        "required_slots_completed": not final_missing,
        "required_tools_called": not missing_tools if config.enforce_required_tools else True,
        "no_early_submit": not penalties["early_submit"],
        "no_repeated_question": not penalties["repeated_question"],
        "no_over_refuse": not penalties["over_refuse"],
    }
    verifier_score = round(sum(checks.values()) / len(checks), 4)
    return {
        "reward_profile": config.name,
        "verifier_score": verifier_score,
        "verifier_checks": checks,
        "missing_tools": missing_tools,
        "penalties": penalties,
    }


def judge_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    message = trajectory["final_decision"].get("message", "")
    length_score = 0.35 if len(message) >= 12 else 0.15
    action_score = 0.25 if trajectory["final_decision"]["type"] in {"SUBMIT", "REFUSE"} else 0
    clarity_score = 0.20 if any(word in message for word in ["符合", "缺", "不足", "提交", "材料"]) else 0.10
    executable_score = 0.20 if any(word in message for word in ["已提交", "补齐", "建议", "暂不能"]) else 0.10
    score = round(min(1.0, length_score + action_score + clarity_score + executable_score), 4)
    return {
        "judge_score": score,
        "judge_dimensions": {
            "length": length_score,
            "action_clarity": action_score,
            "reason_clarity": clarity_score,
            "actionability": executable_score,
        },
    }


def score_trajectory(
    case: GovCase,
    trajectory: dict[str, Any],
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    config = config or reward_profile("hardened")
    verifier = verify_trajectory(case, trajectory, config)
    judge = judge_trajectory(trajectory)
    process_score = _process_score(trajectory)
    penalty_sum = sum(
        config.penalty_weights[name] * value
        for name, value in verifier["penalties"].items()
    )
    step_penalty = config.step_cost * len(trajectory.get("steps", []))
    reward = (
        config.verifier_weight * verifier["verifier_score"]
        + config.judge_weight * judge["judge_score"]
        + config.process_weight * process_score
        - penalty_sum
        - step_penalty
    )
    reward = round(max(0.0, min(1.0, reward)), 4)
    verdict = "good" if reward >= 0.70 else "bad" if reward < 0.45 else "mixed"
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "case_id": trajectory["case_id"],
        "policy_name": trajectory["policy_name"],
        **verifier,
        **judge,
        "process_score": process_score,
        "step_penalty": round(step_penalty, 4),
        "reward": reward,
        "verdict": verdict,
    }


def score_file(
    cases_dir: Path,
    trajectories_path: Path,
    output_path: Path,
    profile_name: str = "hardened",
) -> Path:
    cases = {case.case_id: case for case in load_cases(cases_dir)}
    trajectories = read_jsonl(trajectories_path)
    config = reward_profile(profile_name)
    scored = [score_trajectory(cases[row["case_id"]], row, config) for row in trajectories]
    write_jsonl(output_path, scored)
    return output_path


def _tool_history(trajectory: dict[str, Any]) -> list[str]:
    tools = []
    for step in trajectory.get("steps", []):
        action_type = step.get("action", {}).get("type")
        if action_type in {"POLICY_SEARCH", "ELIGIBILITY_CHECK", "MATERIAL_CHECK", "RISK_CHECK"}:
            tools.append(action_type)
    return tools


def _process_score(trajectory: dict[str, Any]) -> float:
    tags = trajectory.get("failure_tags", [])
    if not tags:
        return 1.0
    return round(max(0.0, 1.0 - 0.15 * len(tags)), 4)

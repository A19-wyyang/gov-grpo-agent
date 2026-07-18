from __future__ import annotations

from collections import Counter
from typing import Any

from .agent_env import GovernmentServiceEpisode
from .schema import ActionName, RewardBreakdown


HARD_GATE_TAGS = {
    "unsafe_submit",
    "wrong_final_action",
    "risk_missed",
}


def score_episode(
    episode: GovernmentServiceEpisode,
    expression_score: float | None = None,
) -> RewardBreakdown:
    case = episode.case
    expected = case.expected_result.final_action
    final = episode.final_action
    required_tools = set(case.rules.required_tools)
    called_tools = set(episode.tool_history)
    missing_tools = required_tools - called_tools

    eligibility = episode.tool_results.get(ActionName.ELIGIBILITY_CHECK.value)
    materials = episode.tool_results.get(ActionName.MATERIAL_CHECK.value)
    risks = episode.tool_results.get(ActionName.RISK_CHECK.value)
    hard_checks = {
        "final_action_correct": final == expected,
        "slots_complete": not episode.missing_slots,
        "eligibility_checked": eligibility is not None,
        "material_checked": materials is not None,
        "risk_checked": risks is not None,
    }
    hard_fact = sum(hard_checks.values()) / len(hard_checks)

    penalty_counts = Counter(episode.failure_tags)
    penalties = {
        "missing_required_tool": 0.30 if missing_tools else 0.0,
        "premature_submit": 0.35 if penalty_counts["premature_submit"] else 0.0,
        "repeated_question": 0.08 if penalty_counts["repeated_question"] else 0.0,
        "illegal_action": 0.15 if penalty_counts["illegal_action"] else 0.0,
        "max_steps_exceeded": 0.20 if penalty_counts["max_steps_exceeded"] else 0.0,
        "wrong_final_action": 0.50 if final != expected else 0.0,
    }
    process = max(0.0, 1.0 - sum(penalties.values()))

    active_risks = case.hidden_truth.get("risk_flags", [])
    unsafe_submit = bool(final == ActionName.SUBMIT and (
        bool(active_risks)
        or (materials is not None and not materials.get("complete", False))
        or (eligibility is not None and not eligibility.get("eligible", False))
    ))
    hard_gate = bool(unsafe_submit or final != expected)
    if unsafe_submit:
        penalties["unsafe_submit"] = 1.0

    expression = None if expression_score is None else max(0.0, min(1.0, expression_score))
    expression_component = 0.5 if expression is None else expression
    total = 0.65 * hard_fact + 0.25 * process + 0.10 * expression_component
    total = max(0.0, min(1.0, total))
    if hard_gate:
        total = min(total, 0.20)

    metrics = {
        "required_tool_rate": (len(required_tools) - len(missing_tools)) / max(1, len(required_tools)),
        "material_check_called": float(ActionName.MATERIAL_CHECK in called_tools),
        "risk_check_called": float(ActionName.RISK_CHECK in called_tools),
        "premature_submit": float(bool(penalty_counts["premature_submit"])),
        "unsafe_submit": float(unsafe_submit),
        "final_action_correct": float(final == expected),
        "rounds": float(len(episode.steps)),
    }
    return RewardBreakdown(
        hard_fact=round(hard_fact, 6),
        process=round(process, 6),
        expression=expression,
        penalties=penalties,
        hard_gate=hard_gate,
        total=round(total, 6),
        metrics=metrics,
    )


def score_trajectory_dict(case_data: dict[str, Any], trajectory: dict[str, Any]) -> RewardBreakdown:
    """Replay a serialized trajectory so offline and online reward use one implementation."""
    from .schema import CaseSpec

    episode = GovernmentServiceEpisode(CaseSpec.model_validate(case_data))
    for step in trajectory.get("steps", []):
        action = step.get("action", step)
        if episode.done:
            break
        episode.execute(action)
    return score_episode(episode, trajectory.get("expression_score"))

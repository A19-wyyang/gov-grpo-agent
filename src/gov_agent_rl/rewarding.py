from __future__ import annotations

import os
from collections import Counter
from typing import Any

from .agent_env import GovernmentServiceEpisode
from .schema import ActionName, RewardBreakdown


HARD_GATE_TAGS = {
    "unsafe_submit",
    "wrong_final_action",
    "risk_missed",
    "tool_result_conflict",
}

PROCESS_FAILURE_TAGS = {
    "premature_submit",
    "repeated_question",
    "invalid_slot_question",
    "illegal_action",
    "max_steps_exceeded",
    "action_after_done",
    "repeated_tool_call",
    "tool_order_violation",
    "eligibility_before_slots_complete",
}


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes"}


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
    verification_complete = all(
        result is not None for result in (eligibility, materials, risks)
    )
    all_checks_passed = bool(
        verification_complete
        and eligibility.get("eligible", False)
        and materials.get("complete", False)
        and risks.get("passed", False)
    )
    tool_results_support_final = bool(
        verification_complete
        and (
            (final == ActionName.SUBMIT and all_checks_passed)
            or (final == ActionName.REFUSE and not all_checks_passed)
        )
    )
    tool_result_conflict = bool(
        verification_complete
        and final in {ActionName.SUBMIT, ActionName.REFUSE}
        and not tool_results_support_final
    )
    hard_checks = {
        "final_action_correct": final == expected,
        "slots_complete": not episode.missing_slots,
        "eligibility_checked": eligibility is not None,
        "material_checked": materials is not None,
        "risk_checked": risks is not None,
        "tool_results_support_final": tool_results_support_final,
    }
    hard_fact = sum(hard_checks.values()) / len(hard_checks)

    penalty_counts = getattr(episode, "failure_counts", Counter(episode.failure_tags))
    illegal_action_count = int(penalty_counts["illegal_action"])
    trailing_action_count = int(penalty_counts["action_after_done"])
    penalties = {
        "missing_required_tool": (
            _env_float("GOV_MISSING_TOOL_PENALTY", 0.30) if missing_tools else 0.0
        ),
        "premature_submit": 0.35 if penalty_counts["premature_submit"] else 0.0,
        "repeated_question": 0.08 if penalty_counts["repeated_question"] else 0.0,
        "invalid_slot_question": (
            _env_float("GOV_INVALID_SLOT_PENALTY", 0.08)
            if penalty_counts["invalid_slot_question"]
            else 0.0
        ),
        "illegal_action": (
            min(
                1.0,
                _env_float("GOV_ILLEGAL_ACTION_PENALTY", 0.15)
                * illegal_action_count,
            )
        ),
        "max_steps_exceeded": 0.20 if penalty_counts["max_steps_exceeded"] else 0.0,
        "action_after_done": min(
            1.0,
            _env_float("GOV_ACTION_AFTER_DONE_PENALTY", 0.10)
            * trailing_action_count,
        ),
        "tool_result_conflict": 0.50 if tool_result_conflict else 0.0,
        "repeated_tool_call": min(
            0.25,
            _env_float("GOV_REPEATED_TOOL_PENALTY", 0.05)
            * int(penalty_counts["repeated_tool_call"]),
        ),
        "tool_order_violation": min(
            0.50,
            _env_float("GOV_TOOL_ORDER_PENALTY", 0.10)
            * int(penalty_counts["tool_order_violation"]),
        ),
        "eligibility_before_slots_complete": (
            _env_float("GOV_EARLY_ELIGIBILITY_PENALTY", 0.15)
            if penalty_counts["eligibility_before_slots_complete"]
            else 0.0
        ),
        "wrong_final_action": 0.50 if final != expected else 0.0,
    }
    process = max(0.0, 1.0 - sum(penalties.values()))

    # A submit is unsafe whenever the verifier-owned expected decision is not
    # SUBMIT. Deriving this only from called tool results undercounts direct
    # submits that deliberately skip eligibility or material checks.
    unsafe_submit = bool(
        final == ActionName.SUBMIT and expected != ActionName.SUBMIT
    )
    incomplete_final = bool(
        missing_tools and final in {ActionName.SUBMIT, ActionName.REFUSE}
    )
    decision_gate = bool(
        unsafe_submit or final != expected or tool_result_conflict
    )
    process_gate = bool(
        _env_bool("GOV_MISSING_TOOL_HARD_GATE") and incomplete_final
    )
    hard_gate = decision_gate or process_gate
    if unsafe_submit:
        penalties["unsafe_submit"] = 1.0
    required_tool_rate = (
        len(required_tools) - len(missing_tools)
    ) / max(1, len(required_tools))
    process_compliant = bool(
        required_tool_rate >= 1.0
        and not episode.missing_slots
        and tool_results_support_final
        and final == expected
        and not any(penalty_counts[tag] for tag in PROCESS_FAILURE_TAGS)
    )

    expression = None if expression_score is None else max(0.0, min(1.0, expression_score))
    # Expression quality must never rescue a verifier hard failure, even when
    # a non-zero diagnostic gate ceiling is configured.
    expression_component = (
        0.0
        if hard_gate
        else 0.5
        if expression is None
        else expression
    )
    hard_fact_weight = _env_float("GOV_HARD_FACT_WEIGHT", 0.65)
    process_weight = _env_float("GOV_PROCESS_WEIGHT", 0.25)
    expression_weight = _env_float("GOV_EXPRESSION_WEIGHT", 0.10)
    weight_sum = hard_fact_weight + process_weight + expression_weight
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"reward component weights must sum to 1, got {weight_sum}")
    total = (
        hard_fact_weight * hard_fact
        + process_weight * process
        + expression_weight * expression_component
    )
    total = max(0.0, min(1.0, total))
    if decision_gate:
        total = min(total, _env_float("GOV_DECISION_GATE_CEILING", 0.20))
    elif process_gate:
        total = min(total, _env_float("GOV_PROCESS_GATE_CEILING", 0.20))

    metrics = {
        "required_tool_rate": required_tool_rate,
        "material_check_called": float(ActionName.MATERIAL_CHECK in called_tools),
        "risk_check_called": float(ActionName.RISK_CHECK in called_tools),
        "premature_submit": float(bool(penalty_counts["premature_submit"])),
        "unsafe_submit": float(unsafe_submit),
        "missing_required_tool": float(bool(missing_tools)),
        "incomplete_final": float(incomplete_final),
        "decision_gate": float(decision_gate),
        "process_gate": float(process_gate),
        "illegal_action": float(bool(penalty_counts["illegal_action"])),
        "illegal_action_count": float(illegal_action_count),
        "illegal_action_attempt_rate": (
            illegal_action_count / max(1, episode.action_attempts)
        ),
        "trailing_action_count": float(trailing_action_count),
        "trailing_action_rate": (
            trailing_action_count
            / max(1, episode.action_attempts + trailing_action_count)
        ),
        "invalid_slot_question": float(bool(penalty_counts["invalid_slot_question"])),
        "max_steps_exceeded": float(bool(penalty_counts["max_steps_exceeded"])),
        "final_action_correct": float(final == expected),
        "tool_results_support_final": float(tool_results_support_final),
        "tool_result_conflict": float(tool_result_conflict),
        "repeated_tool_call": float(bool(penalty_counts["repeated_tool_call"])),
        "repeated_tool_call_count": float(
            penalty_counts["repeated_tool_call"]
        ),
        "tool_order_violation": float(
            bool(penalty_counts["tool_order_violation"])
        ),
        "eligibility_before_slots_complete": float(
            bool(penalty_counts["eligibility_before_slots_complete"])
        ),
        "process_compliant": float(process_compliant),
        "rounds": float(episode.action_attempts),
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
    serialized_actions = trajectory.get("attempts")
    uses_attempts = serialized_actions is not None
    if not uses_attempts:
        serialized_actions = trajectory.get("steps", [])
    for step in serialized_actions:
        action = step if uses_attempts else step.get("action", step)
        episode.execute(action)
    return score_episode(episode, trajectory.get("expression_score"))

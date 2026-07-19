from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gov_agent_rl.judge import RUBRIC, judge_expression_detailed
from gov_agent_rl.rewarding import score_trajectory_dict
from gov_agent_rl.schema import ActionName


def _find_case(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if (
            "case_id" in value
            and "hidden_truth" in value
            and "rules" in value
        ):
            return value
        for nested in value.values():
            case = _find_case(nested)
            if case is not None:
                return case
    elif isinstance(value, str):
        try:
            return _find_case(json.loads(value))
        except json.JSONDecodeError:
            return None
    return None


def _tool_actions(text: str) -> list[dict[str, Any]]:
    """Recover government-service calls from a decoded multi-turn response."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "name" in payload and "arguments" in payload:
            if payload.get("name") != "government_service":
                # Preserve malformed/unknown calls so replay penalizes them
                # instead of silently dropping them from the trajectory.
                calls.append({"action": "__INVALID_TOOL_NAME__"})
                continue
            action = payload.get("arguments")
        elif isinstance(payload.get("function"), dict):
            if payload["function"].get("name") != "government_service":
                calls.append({"action": "__INVALID_TOOL_NAME__"})
                continue
            action = payload["function"].get("arguments")
        else:
            continue
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                continue
        if isinstance(action, dict) and "action" in action:
            calls.append(action)
    return calls


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> float | dict[str, Any]:
    """Replay a rollout and return the verifier-owned environment reward.

    The optional Qwen Judge contributes only the expression component inside
    ``score_trajectory_dict``; hard facts and hard gates remain deterministic.
    """
    if data_source != "gov_agent_rl":
        return 0.0
    text = solution_str.strip()
    if not text:
        return 0.0
    case = _find_case(extra_info or {})
    if case is None:
        return 0.0

    actions = _tool_actions(solution_str)
    final_action = ""
    final_message = ""
    for action in reversed(actions):
        if action.get("action") in {
            ActionName.SUBMIT.value,
            ActionName.REFUSE.value,
        }:
            final_action = str(action["action"])
            final_message = str(action.get("message", ""))
            break
    judge_result = judge_expression_detailed(
        user_request=str(case.get("user_request", "")),
        final_action=final_action,
        message=final_message,
        cache_path=Path(
            os.getenv("GOV_JUDGE_CACHE", "runs/judge/qwen_expression.sqlite3")
        ),
    )
    judge_score = None if judge_result is None else judge_result[0]
    judge_fallback_score = float(os.getenv("GOV_JUDGE_FAILURE_SCORE", "0.0"))
    expression_score = judge_score if judge_score is not None else judge_fallback_score
    judge_payload = {} if judge_result is None else judge_result[1]
    judge_dimensions = judge_payload.get("dimensions", {})
    judge_metrics: dict[str, float] = {}
    for name in RUBRIC:
        item = judge_dimensions.get(name)
        raw_score = item.get("score") if isinstance(item, dict) else item
        try:
            judge_metrics[f"judge_{name}"] = float(raw_score) / 4.0
        except (TypeError, ValueError):
            judge_metrics[f"judge_{name}"] = -1.0
    breakdown = score_trajectory_dict(
        case,
        {
            "steps": [{"action": action} for action in actions],
            "expression_score": expression_score,
        },
    )
    return {
        "score": breakdown.total,
        "case_id": case["case_id"],
        "scenario_type": case.get("scenario_type", "unknown"),
        "environment_reward": breakdown.total,
        "judge_score": -1.0 if judge_score is None else judge_score,
        "judge_used": float(judge_score is not None),
        "judge_fallback_used": float(judge_score is None),
        **judge_metrics,
        "hard_gate": float(breakdown.hard_gate),
        "parsed_action_count": len(actions),
        **breakdown.metrics,
    }

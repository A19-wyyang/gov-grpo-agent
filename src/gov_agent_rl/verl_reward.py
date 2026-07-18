from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gov_agent_rl.judge import judge_expression
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
        if payload.get("name") == "government_service":
            action = payload.get("arguments")
        elif payload.get("function", {}).get("name") == "government_service":
            action = payload["function"].get("arguments")
        else:
            continue
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                continue
        if (
            isinstance(action, dict)
            and action.get("action") in {item.value for item in ActionName}
        ):
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
    judge_score = judge_expression(
        user_request=str(case.get("user_request", "")),
        final_action=final_action,
        message=final_message,
        cache_path=Path(
            os.getenv("GOV_JUDGE_CACHE", "runs/judge/qwen_expression.sqlite3")
        ),
    )
    breakdown = score_trajectory_dict(
        case,
        {
            "steps": [{"action": action} for action in actions],
            "expression_score": judge_score,
        },
    )
    return {
        "score": breakdown.total,
        "case_id": case["case_id"],
        "scenario_type": case.get("scenario_type", "unknown"),
        "environment_reward": breakdown.total,
        "judge_score": -1.0 if judge_score is None else judge_score,
        "judge_used": float(judge_score is not None),
        "hard_gate": float(breakdown.hard_gate),
        "parsed_action_count": len(actions),
        **breakdown.metrics,
    }

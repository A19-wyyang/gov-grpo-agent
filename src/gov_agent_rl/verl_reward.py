from __future__ import annotations

import json
from typing import Any

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
    """Completion-only reward; factual trajectory reward is owned by the tool.

    This deliberately has a small range so fluent text cannot overwhelm the
    stateful verifier. It checks format/action clarity only.
    """
    if data_source != "gov_agent_rl":
        return 0.0
    text = solution_str.strip()
    if not text:
        return 0.0
    score = 0.02
    if len(text) >= 16:
        score += 0.03
    if any(word in text for word in ("提交", "暂不能", "补齐", "转人工")):
        score += 0.03
    expected = str(ground_truth.get("final_action", ""))
    if expected and expected in text.upper():
        score += 0.02
    expression_score = min(0.10, score)
    case = _find_case(extra_info or {})
    if case is None:
        return expression_score

    actions = _tool_actions(solution_str)
    breakdown = score_trajectory_dict(
        case,
        {
            "steps": [{"action": action} for action in actions],
            "expression_score": expression_score / 0.10,
        },
    )
    return {
        "score": expression_score,
        "case_id": case["case_id"],
        "scenario_type": case.get("scenario_type", "unknown"),
        "environment_reward": breakdown.total,
        "hard_gate": float(breakdown.hard_gate),
        "parsed_action_count": len(actions),
        **breakdown.metrics,
    }

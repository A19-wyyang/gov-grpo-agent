#!/usr/bin/env python3
"""Replay saved rollouts with one common verifier/reward definition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gov_agent_rl.rewarding import score_trajectory_dict
from gov_agent_rl.fingerprints import case_fingerprint
from gov_agent_rl.judge import RUBRIC
from gov_agent_rl.verl_reward import _tool_actions


REWARD_ENV_NAMES = (
    "GOV_MISSING_TOOL_PENALTY",
    "GOV_MISSING_TOOL_HARD_GATE",
    "GOV_DECISION_GATE_CEILING",
    "GOV_PROCESS_GATE_CEILING",
    "GOV_INVALID_SLOT_PENALTY",
    "GOV_ILLEGAL_ACTION_PENALTY",
    "GOV_ACTION_AFTER_DONE_PENALTY",
    "GOV_REPEATED_TOOL_PENALTY",
    "GOV_TOOL_ORDER_PENALTY",
    "GOV_EARLY_ELIGIBILITY_PENALTY",
    "GOV_HARD_FACT_WEIGHT",
    "GOV_PROCESS_WEIGHT",
    "GOV_EXPRESSION_WEIGHT",
    "GOV_JUDGE_FAILURE_SCORE",
)


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {
            record["case_id"]: record
            for line in handle
            if line.strip()
            for record in [json.loads(line)]
        }


def rescore_record(
    record: dict[str, Any],
    case: dict[str, Any],
    allow_missing_case_fingerprint: bool = True,
) -> dict[str, Any]:
    expected_fingerprint = case_fingerprint(case)
    stored_fingerprint = record.get("case_fingerprint")
    if stored_fingerprint is None and not allow_missing_case_fingerprint:
        raise ValueError(
            f"rollout {record.get('case_id')} has no case_fingerprint; "
            "use --allow-missing-case-fingerprint only for an exact archived "
            "legacy dataset"
        )
    if (
        stored_fingerprint is not None
        and str(stored_fingerprint) != expected_fingerprint
    ):
        raise ValueError(
            f"case fingerprint mismatch for {record.get('case_id')}: "
            f"rollout={stored_fingerprint} cases={expected_fingerprint}"
        )
    judge_used = float(record.get("judge_used", 0.0)) > 0
    stored_judge = float(record.get("judge_score", -1.0))
    expression_score = (
        stored_judge
        if judge_used and stored_judge >= 0
        else float(os.getenv("GOV_JUDGE_FAILURE_SCORE", "0.0"))
    )
    actions = _tool_actions(str(record.get("output", "")))
    final_actions = [
        action
        for action in actions
        if action.get("action") in {"SUBMIT", "REFUSE"}
    ]
    parsed_empty_message = bool(
        final_actions
        and not str(final_actions[-1].get("message", "")).strip()
    )
    breakdown = score_trajectory_dict(
        case,
        {
            "steps": [{"action": action} for action in actions],
            "expression_score": expression_score,
        },
    )
    rescored = dict(record)
    rescored["source_environment_reward"] = record.get("environment_reward")
    rescored["source_judge_used"] = record.get("judge_used")
    rescored["source_judge_score"] = record.get("judge_score")
    rescored["case_fingerprint"] = expected_fingerprint
    rescored["scenario_type"] = case.get("scenario_type", "unknown")
    rescored["matter_id"] = case.get("matter_id", "unknown")
    rescored["split"] = case.get("split", "unknown")
    rescored["environment_reward"] = breakdown.total
    rescored["score"] = breakdown.total
    rescored["hard_gate"] = float(breakdown.hard_gate)
    rescored["parsed_action_count"] = len(actions)
    hard_gate = bool(breakdown.hard_gate)
    empty_message = bool(
        parsed_empty_message
        or float(record.get("judge_empty_message", 0.0)) > 0
    )
    rescored["judge_skipped_hard_gate"] = float(hard_gate)
    rescored["judge_empty_message"] = float(empty_message)
    rescored["judge_fallback_used"] = float(
        not judge_used and not hard_gate and not empty_message
    )
    effective_judge_used = bool(judge_used and not hard_gate)
    rescored["judge_used"] = float(effective_judge_used)
    rescored["judge_score"] = (
        stored_judge if effective_judge_used else -1.0
    )
    if not effective_judge_used:
        for name in RUBRIC:
            rescored[f"judge_{name}"] = -1.0
    rescored.update(breakdown.metrics)
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-case-fingerprint",
        action="store_true",
        help="Allow pre-fingerprint rollouts; valid only with their exact archived legacy cases.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing_fingerprint_count = 0
    missing: set[str] = set()
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record.get("case_id", ""))
            case = cases.get(case_id)
            if case is None:
                missing.add(case_id)
                continue
            if record.get("case_fingerprint") is None:
                missing_fingerprint_count += 1
            target.write(
                json.dumps(
                    rescore_record(
                        record,
                        case,
                        allow_missing_case_fingerprint=args.allow_missing_case_fingerprint,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if missing:
        raise ValueError(f"missing {len(missing)} cases: {sorted(missing)[:5]}")
    metadata = {
        "source": str(args.input),
        "cases": str(args.cases),
        "records": count,
        "allow_missing_case_fingerprint": args.allow_missing_case_fingerprint,
        "missing_case_fingerprint_records": missing_fingerprint_count,
        "reward_environment": {name: os.getenv(name) for name in REWARD_ENV_NAMES},
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rescored {count} rollouts with common verifier: {args.output}")


if __name__ == "__main__":
    main()

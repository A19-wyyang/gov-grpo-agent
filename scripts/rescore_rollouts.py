#!/usr/bin/env python3
"""Replay saved rollouts with one common verifier/reward definition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from gov_agent_rl.rewarding import score_trajectory_dict
from gov_agent_rl.verl_reward import _tool_actions


REWARD_ENV_NAMES = (
    "GOV_MISSING_TOOL_PENALTY",
    "GOV_MISSING_TOOL_HARD_GATE",
    "GOV_DECISION_GATE_CEILING",
    "GOV_PROCESS_GATE_CEILING",
    "GOV_INVALID_SLOT_PENALTY",
    "GOV_ILLEGAL_ACTION_PENALTY",
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


def rescore_record(record: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    judge_used = float(record.get("judge_used", 0.0)) > 0
    stored_judge = float(record.get("judge_score", -1.0))
    expression_score = (
        stored_judge
        if judge_used and stored_judge >= 0
        else float(os.getenv("GOV_JUDGE_FAILURE_SCORE", "0.0"))
    )
    actions = _tool_actions(str(record.get("output", "")))
    breakdown = score_trajectory_dict(
        case,
        {
            "steps": [{"action": action} for action in actions],
            "expression_score": expression_score,
        },
    )
    rescored = dict(record)
    rescored["source_environment_reward"] = record.get("environment_reward")
    rescored["environment_reward"] = breakdown.total
    rescored["score"] = breakdown.total
    rescored["hard_gate"] = float(breakdown.hard_gate)
    rescored["parsed_action_count"] = len(actions)
    rescored["judge_fallback_used"] = float(not judge_used)
    rescored.update(breakdown.metrics)
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
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
            target.write(json.dumps(rescore_record(record, case), ensure_ascii=False) + "\n")
            count += 1
    if missing:
        raise ValueError(f"missing {len(missing)} cases: {sorted(missing)[:5]}")
    metadata = {
        "source": str(args.input),
        "cases": str(args.cases),
        "records": count,
        "reward_environment": {name: os.getenv(name) for name in REWARD_ENV_NAMES},
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rescored {count} rollouts with common verifier: {args.output}")


if __name__ == "__main__":
    main()

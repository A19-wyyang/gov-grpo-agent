#!/usr/bin/env python3
"""Check that verifier tool results support every generated expected decision."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gov_agent_rl.agent_env import GovernmentServiceEpisode
from gov_agent_rl.schema import ActionName, CaseSpec


def derive_action_from_tools(case: CaseSpec) -> tuple[ActionName, dict[str, bool]]:
    episode = GovernmentServiceEpisode(case)
    for slot in case.rules.required_slots:
        if slot not in episode.known_slots:
            episode.execute({"action": "ASK_USER", "slot": slot})
    episode.execute({"action": "POLICY_SEARCH", "query": case.title})
    eligibility = episode.execute({"action": "ELIGIBILITY_CHECK"})
    materials = episode.execute({"action": "MATERIAL_CHECK"})
    risks = episode.execute({"action": "RISK_CHECK"})
    checks = {
        "eligible": bool(eligibility["eligible"]),
        "materials_complete": bool(materials["complete"]),
        "risk_passed": bool(risks["passed"]),
    }
    derived = (
        ActionName.SUBMIT if all(checks.values()) else ActionName.REFUSE
    )
    return derived, checks


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    conflicts = []
    for record in records:
        case = CaseSpec.model_validate(record)
        derived, checks = derive_action_from_tools(case)
        if derived != case.expected_result.final_action:
            conflicts.append(
                {
                    "case_id": case.case_id,
                    "matter_id": case.matter_id,
                    "scenario_type": case.scenario_type,
                    "expected": case.expected_result.final_action.value,
                    "derived": derived.value,
                    "checks": checks,
                }
            )
    return {
        "count": len(records),
        "conflict_count": len(conflicts),
        "conflicts_by_matter": dict(
            Counter(item["matter_id"] for item in conflicts)
        ),
        "conflicts_by_scenario": dict(
            Counter(item["scenario_type"] for item in conflicts)
        ),
        "conflicts": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-consistent", action="store_true")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_records(records)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_consistent and report["conflict_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

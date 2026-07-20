from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _process_compliant(row: dict[str, Any]) -> bool:
    return float(
        row.get(
            "process_compliant",
            float(row.get("required_tool_rate", 0.0)) >= 1.0,
        )
    ) >= 1.0


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)

    rewards = [
        float(
            row.get(
                "environment_reward",
                row.get("reward", row.get("score", 0.0)),
            )
        )
        for row in rows
    ]
    pass_flags = [float(row.get("final_action_correct", 0.0)) for row in rows]
    case_pass = {
        case_id: any(float(item.get("final_action_correct", 0.0)) > 0 for item in items)
        for case_id, items in grouped.items()
    }
    case_safe_pass = {
        case_id: any(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            for item in items
        )
        for case_id, items in grouped.items()
    }
    case_process_pass = {
        case_id: any(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            and _process_compliant(item)
            for item in items
        )
        for case_id, items in grouped.items()
    }
    scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_rows[str(row.get("scenario_type", "unknown"))].append(row)

    def row_mean(key: str) -> float:
        return round(mean(float(row.get(key, 0.0)) for row in rows), 6)

    def output_diversity(
        case_groups: dict[str, list[dict[str, Any]]],
    ) -> tuple[float, float]:
        unique_rates = []
        identical_groups = []
        for items in case_groups.values():
            unique = {
                " ".join(str(item.get("output", "")).split())
                for item in items
            }
            unique_rates.append(len(unique) / len(items))
            identical_groups.append(
                float(len(items) > 1 and len(unique) == 1)
            )
        return mean(unique_rates), mean(identical_groups)

    unique_output_rate, identical_output_group_rate = output_diversity(grouped)

    judged_scores = [
        float(row["judge_score"])
        for row in rows
        if float(row.get("judge_used", 0.0)) > 0 and float(row.get("judge_score", -1.0)) >= 0
    ]

    def _scenario_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        scenario_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            scenario_grouped[str(item["case_id"])].append(item)
        scenario_pass_at_k = mean(
            any(float(item.get("final_action_correct", 0.0)) > 0 for item in case_items)
            for case_items in scenario_grouped.values()
        )
        scenario_safe_at_k = mean(
            any(
                float(item.get("final_action_correct", 0.0)) > 0
                and float(item.get("hard_gate", 0.0)) == 0
                for item in case_items
            )
            for case_items in scenario_grouped.values()
        )
        scenario_safe_at_1 = mean(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            for item in items
        )
        scenario_process_at_1 = mean(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            and _process_compliant(item)
            for item in items
        )
        scenario_process_at_k = mean(
            any(
                float(item.get("final_action_correct", 0.0)) > 0
                and float(item.get("hard_gate", 0.0)) == 0
                and _process_compliant(item)
                for item in case_items
            )
            for case_items in scenario_grouped.values()
        )
        scenario_unique_output_rate, scenario_identical_group_rate = (
            output_diversity(scenario_grouped)
        )
        return {
            "count": len(items),
            "case_count": len(scenario_grouped),
            "pass_at_1": round(
                mean(float(item.get("final_action_correct", 0.0)) for item in items),
                6,
            ),
            "pass_at_k": round(scenario_pass_at_k, 6),
            "safe_pass_at_k": round(scenario_safe_at_k, 6),
            "safe_pass_at_1": round(scenario_safe_at_1, 6),
            "process_pass_at_1": round(scenario_process_at_1, 6),
            "process_pass_at_k": round(scenario_process_at_k, 6),
            "unique_output_rate": round(
                scenario_unique_output_rate, 6
            ),
            "identical_output_group_rate": round(
                scenario_identical_group_rate, 6
            ),
            "mean_reward": round(
                mean(
                    float(
                        item.get(
                            "environment_reward",
                            item.get("reward", item.get("score", 0.0)),
                        )
                    )
                    for item in items
                ),
                6,
            ),
            "unsafe_submit_rate": round(
                mean(float(item.get("unsafe_submit", 0.0)) for item in items),
                6,
            ),
            "tool_result_conflict_rate": round(
                mean(
                    float(item.get("tool_result_conflict", 0.0))
                    for item in items
                ),
                6,
            ),
        }

    metrics = {
        "count": len(rows),
        "case_count": len(grouped),
        "rollouts_per_case": sorted({len(items) for items in grouped.values()}),
        "mean_reward": round(mean(rewards), 6),
        "judge_coverage": row_mean("judge_used"),
        "mean_judge_score": (
            round(mean(judged_scores), 6) if judged_scores else None
        ),
        "pass_at_1": round(mean(pass_flags), 6),
        "pass_at_k": round(mean(float(value) for value in case_pass.values()), 6),
        "safe_pass_at_1": round(
            mean(
                float(row.get("final_action_correct", 0.0)) > 0
                and float(row.get("hard_gate", 0.0)) == 0
                for row in rows
            ),
            6,
        ),
        "safe_pass_at_k": round(
            mean(float(value) for value in case_safe_pass.values()), 6
        ),
        "process_pass_at_1": round(
            mean(
                float(row.get("final_action_correct", 0.0)) > 0
                and float(row.get("hard_gate", 0.0)) == 0
                and _process_compliant(row)
                for row in rows
            ),
            6,
        ),
        "process_pass_at_k": round(
            mean(float(value) for value in case_process_pass.values()), 6
        ),
        "unique_output_rate": round(unique_output_rate, 6),
        "identical_output_group_rate": round(
            identical_output_group_rate, 6
        ),
        "hard_gate_rate": row_mean("hard_gate"),
        "required_tool_rate": row_mean("required_tool_rate"),
        "process_compliance_rate": round(
            mean(float(_process_compliant(row)) for row in rows), 6
        ),
        "repeated_tool_call_rate": row_mean("repeated_tool_call"),
        "tool_order_violation_rate": row_mean("tool_order_violation"),
        "early_eligibility_rate": row_mean(
            "eligibility_before_slots_complete"
        ),
        "material_check_rate": row_mean("material_check_called"),
        "risk_check_rate": row_mean("risk_check_called"),
        "premature_submit_rate": row_mean("premature_submit"),
        "unsafe_submit_rate": row_mean("unsafe_submit"),
        "tool_result_conflict_rate": row_mean("tool_result_conflict"),
        "mean_rounds": row_mean("rounds"),
        "mean_parsed_actions": row_mean("parsed_action_count"),
        "scenario_counts": dict(Counter(row.get("scenario_type", "unknown") for row in rows)),
        "scenario_metrics": {
            scenario: _scenario_metrics(items)
            for scenario, items in sorted(scenario_rows.items())
        },
    }
    return metrics


def evaluate_jsonl(input_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = evaluate_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics

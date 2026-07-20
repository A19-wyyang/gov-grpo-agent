#!/usr/bin/env python3
"""Select a GRPO checkpoint by common-verifier safety/process metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scripts.rescore_rollouts import load_cases, rescore_record


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (center - margin) / denominator


def group_complete_records(
    records: list[dict[str, Any]],
    expected_case_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        grouped.setdefault(case_id, []).append(record)
    actual = set(grouped)
    missing = sorted(expected_case_ids - actual)
    unknown = sorted(actual - expected_case_ids)
    if missing or unknown:
        raise ValueError(
            f"incomplete case coverage: missing={missing[:5]} "
            f"unknown={unknown[:5]}"
        )
    rollout_counts = {len(items) for items in grouped.values()}
    if len(rollout_counts) != 1:
        raise ValueError(
            f"non-uniform rollouts per case: {sorted(rollout_counts)}"
        )
    rollouts_per_case = next(iter(rollout_counts), 0)
    if rollouts_per_case <= 0:
        raise ValueError("validation contains no rollouts")
    return grouped, rollouts_per_case


def summarize_rescored(
    records: list[dict[str, Any]],
    expected_case_ids: set[str],
) -> dict[str, float]:
    grouped, rollouts_per_case = group_complete_records(records, expected_case_ids)
    case_count = len(grouped)
    def safe_record(record: dict[str, Any]) -> bool:
        return bool(
            float(record.get("final_action_correct", 0.0)) > 0
            and float(record.get("hard_gate", 0.0)) == 0
        )

    def process_record(record: dict[str, Any]) -> bool:
        return bool(
            safe_record(record)
            and float(
                record.get(
                    "process_compliant",
                    float(record.get("required_tool_rate", 0.0)) >= 1.0,
                )
            )
            >= 1.0
        )

    safe_at_1_count = sum(safe_record(record) for record in records)
    process_at_1_count = sum(process_record(record) for record in records)
    safe_at_k_count = sum(
        any(safe_record(record) for record in items)
        for items in grouped.values()
    )
    process_at_k_count = sum(
        any(process_record(record) for record in items)
        for items in grouped.values()
    )

    scenario_cases: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for case_id, items in grouped.items():
        scenarios = {
            str(record.get("scenario_type", "unknown")) for record in items
        }
        if len(scenarios) != 1:
            raise ValueError(
                f"case {case_id} has inconsistent scenario labels: "
                f"{sorted(scenarios)}"
            )
        scenario = next(iter(scenarios))
        scenario_cases.setdefault(scenario, {})[case_id] = items
    scenario_process_at_1 = {
        scenario: mean(
            float(process_record(record))
            for items in cases.values()
            for record in items
        )
        for scenario, cases in scenario_cases.items()
    }
    scenario_process_at_1_lcb95 = {
        scenario: wilson_lower(
            sum(
                process_record(record)
                for items in cases.values()
                for record in items
            ),
            sum(len(items) for items in cases.values()),
        )
        for scenario, cases in scenario_cases.items()
    }
    scenario_process_at_k = {
        scenario: sum(
            any(process_record(record) for record in items)
            for items in cases.values()
        )
        / len(cases)
        for scenario, cases in scenario_cases.items()
    }
    scenario_process_at_k_lcb95 = {
        scenario: wilson_lower(
            sum(
                any(process_record(record) for record in items)
                for items in cases.values()
            ),
            len(cases),
        )
        for scenario, cases in scenario_cases.items()
    }
    process_at_1 = process_at_1_count / len(records)
    safe_at_1 = safe_at_1_count / len(records)
    process_at_k = process_at_k_count / case_count
    safe_at_k = safe_at_k_count / case_count
    return {
        "cases": float(case_count),
        "rollouts": float(len(records)),
        "rollouts_per_case": float(rollouts_per_case),
        "process_pass_at_1": process_at_1,
        "process_pass_at_1_lcb95": wilson_lower(
            process_at_1_count, len(records)
        ),
        "process_pass_at_k": process_at_k,
        "process_pass_at_k_lcb95": wilson_lower(
            process_at_k_count, case_count
        ),
        "safe_pass_at_1": safe_at_1,
        "safe_pass_at_1_lcb95": wilson_lower(
            safe_at_1_count, len(records)
        ),
        "safe_pass_at_k": safe_at_k,
        "safe_pass_at_k_lcb95": wilson_lower(safe_at_k_count, case_count),
        "scenario_count": float(len(scenario_cases)),
        "worst_scenario_process_pass_at_1": min(
            scenario_process_at_1.values()
        ),
        "worst_scenario_process_pass_at_1_lcb95": min(
            scenario_process_at_1_lcb95.values()
        ),
        "worst_scenario_process_pass_at_k": min(
            scenario_process_at_k.values()
        ),
        "worst_scenario_process_pass_at_k_lcb95": min(
            scenario_process_at_k_lcb95.values()
        ),
        "scenario_process_pass_at_1_std": (
            pstdev(scenario_process_at_1.values())
            if len(scenario_process_at_1) > 1
            else 0.0
        ),
        # Backward-compatible aliases retain their historical @k meaning.
        "process_success": process_at_k,
        "process_success_lcb95": wilson_lower(
            process_at_k_count, case_count
        ),
        "safe_success": safe_at_k,
        "safe_success_lcb95": wilson_lower(safe_at_k_count, case_count),
        "unsafe_submit": mean(
            float(record.get("unsafe_submit", 0.0)) for record in records
        ),
        "illegal_action_attempt_rate": mean(
            float(record.get("illegal_action_attempt_rate", 0.0))
            for record in records
        ),
        "final_action_correct": mean(
            float(record.get("final_action_correct", 0.0)) for record in records
        ),
        "required_tool_rate": mean(
            float(record.get("required_tool_rate", 0.0)) for record in records
        ),
        "process_compliant": mean(
            float(
                record.get(
                    "process_compliant",
                    float(record.get("required_tool_rate", 0.0)) >= 1.0,
                )
            )
            for record in records
        ),
        "repeated_tool_call": mean(
            float(record.get("repeated_tool_call", 0.0)) for record in records
        ),
        "tool_order_violation": mean(
            float(record.get("tool_order_violation", 0.0)) for record in records
        ),
        "environment_reward": mean(
            float(record["environment_reward"]) for record in records
        ),
    }


def selection_key(row: dict[str, float]) -> tuple[float, ...]:
    return (
        row["worst_scenario_process_pass_at_1_lcb95"],
        row["process_pass_at_1_lcb95"],
        row["safe_pass_at_1_lcb95"],
        -row["unsafe_submit"],
        -row["illegal_action_attempt_rate"],
        row["process_pass_at_k_lcb95"],
        row["safe_pass_at_k_lcb95"],
        row["final_action_correct"],
        row["environment_reward"],
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_selection(rows: list[dict[str, float]], best_step: int, output: Path) -> None:
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 25), "Reward-v2 checkpoint selection", fill="#111827", font=_font(28, True))
    draw.text(
        (42, 66),
        f"Best step: {best_step} (worst-scenario process pass@1 LCB is primary)",
        fill="#4B5563",
        font=_font(17),
    )
    left, top, right, bottom = 90, 125, 1220, 635
    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2)
    for index in range(6):
        y = bottom - index * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((35, y - 10), f"{index / 5:.1f}", fill="#6B7280", font=_font(14))
    steps = [int(row["step"]) for row in rows]
    min_step, max_step = min(steps), max(steps)
    x_for = lambda step: left + (step - min_step) / max(1, max_step - min_step) * (right - left)
    series = (
        (
            "worst-scenario process@1 LCB",
            "worst_scenario_process_pass_at_1_lcb95",
            "#D97706",
        ),
        ("process-safe pass@1", "process_pass_at_1", "#16A34A"),
        ("safe pass@1", "safe_pass_at_1", "#2563EB"),
        ("unsafe submit", "unsafe_submit", "#DC2626"),
    )
    for legend_index, (label, key, color) in enumerate(series):
        points = [(x_for(int(row["step"])), bottom - float(row[key]) * (bottom - top)) for row in rows]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        lx = 70 + legend_index * 300
        draw.line((lx, 680, lx + 32, 680), fill=color, width=4)
        draw.text((lx + 42, 668), label, fill="#374151", font=_font(15))
    for step in steps:
        x = x_for(step)
        draw.text((x - 10, bottom + 15), str(step), fill="#4B5563", font=_font(14))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    common_paths: dict[int, Path] = {}
    skipped_steps: list[dict[str, object]] = []
    expected_case_ids = set(cases)
    for validation_path in sorted(
        args.validation_dir.glob("*.jsonl"), key=lambda path: int(path.stem)
    ):
        step = int(validation_path.stem)
        if step <= 0 or not (args.checkpoint_root / f"global_step_{step}").is_dir():
            continue
        records = [json.loads(line) for line in validation_path.open(encoding="utf-8") if line.strip()]
        try:
            group_complete_records(records, expected_case_ids)
        except ValueError as exc:
            skipped_steps.append({"step": step, "reason": str(exc)})
            continue
        rescored = [rescore_record(record, cases[str(record["case_id"])]) for record in records]
        common_path = args.output_dir / f"candidate_step_{step}_common_reward.jsonl"
        common_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rescored),
            encoding="utf-8",
        )
        common_paths[step] = common_path
        rows.append({"step": float(step), **summarize_rescored(rescored, expected_case_ids)})
    if not rows:
        raise ValueError("no validation step has a matching checkpoint")
    best = max(
        rows,
        key=selection_key,
    )
    best_step = int(best["step"])
    with (args.output_dir / "checkpoint_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "best_step": best_step,
        "best_checkpoint": str(args.checkpoint_root / f"global_step_{best_step}"),
        "best_common_rollouts": str(common_paths[best_step]),
        "selection_order": [
            "worst_scenario_process_pass_at_1_lcb95 desc",
            "process_pass_at_1_lcb95 desc",
            "safe_pass_at_1_lcb95 desc",
            "unsafe_submit asc",
            "illegal_action_attempt_rate asc",
            "process_pass_at_k_lcb95 desc",
            "safe_pass_at_k_lcb95 desc",
            "final_action_correct desc",
            "environment_reward desc",
        ],
        "metrics": rows,
        "skipped_steps": skipped_steps,
    }
    (args.output_dir / "best_checkpoint.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    draw_selection(rows, best_step, args.output_dir / "checkpoint_selection.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two GRPO validation snapshots and export a compact A/B report."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


METRICS = (
    ("environment_reward", "Reward", True),
    ("final_action_correct", "Final action", True),
    ("tool_results_support_final", "Tool-result consistency", True),
    ("tool_result_conflict", "Tool-result conflict", False),
    ("required_tool_rate", "Required tools", True),
    ("process_compliant", "Process compliant", True),
    ("repeated_tool_call", "Repeated tool call", False),
    ("tool_order_violation", "Tool-order violation", False),
    (
        "eligibility_before_slots_complete",
        "Eligibility before slots complete",
        False,
    ),
    ("material_check_called", "Material check", True),
    ("risk_check_called", "Risk check", True),
    ("safe_success_at_1", "Safe success@1", True),
    ("process_success_at_1", "Process success@1", True),
    ("safe_success_at_k", "Safe success@k", True),
    ("process_success_at_k", "Process success@k", True),
    ("unique_output_rate", "Within-case unique outputs", True),
    ("identical_output_group_rate", "Identical-output groups", False),
    ("hard_gate", "Hard-gate rate", False),
    ("unsafe_submit", "Unsafe submit", False),
    ("missing_tool_final_rate", "Missing-tool final", False),
    ("illegal_action_attempt_rate", "Illegal action attempts", False),
    ("trailing_action_rate", "Actions after final", False),
    ("invalid_tool_name_rate", "Invalid tool name", False),
    ("tool_call_format_error_rate", "Malformed tool call", False),
    ("judge_used", "Judge coverage", True),
)

SCENARIO_METRICS = (
    ("process_success_at_1", "Process-safe success@1", True),
    ("safe_success_at_1", "Safe success@1", True),
    ("process_success_at_k", "Process-safe success", True),
    ("safe_success_at_k", "Safe success", True),
    ("final_action_correct", "Final action", True),
    ("tool_result_conflict", "Tool-result conflict", False),
    ("hard_gate", "Hard-gate rate", False),
    ("unsafe_submit", "Unsafe submit", False),
)


def read_step(path: Path, step: int) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(float(row["step"])) == step]
    if not matches:
        available = ", ".join(row["step"] for row in rows) or "none"
        raise ValueError(f"step {step} not found in {path}; available: {available}")
    return {key: float(value) for key, value in matches[-1].items() if value != ""}


def read_case_metrics(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                grouped[str(record["case_id"])].append(record)

    result: dict[str, dict[str, float]] = {}
    raw_metrics = {
        key for key, _, _ in METRICS
        if key not in {
            "safe_success_at_k",
            "safe_success_at_1",
            "process_success_at_k",
            "process_success_at_1",
            "missing_tool_final_rate",
            "invalid_tool_name_rate",
            "tool_call_format_error_rate",
            "unique_output_rate",
            "identical_output_group_rate",
        }
    }
    for case_id, records in grouped.items():
        values: dict[str, float] = {}
        for key in raw_metrics:
            present = [float(record[key]) for record in records if key in record]
            if present:
                values[key] = mean(present)
        safe_flags = [
            float(
                float(record.get("final_action_correct", 0.0)) > 0
                and float(record.get("hard_gate", 0.0)) == 0
            )
            for record in records
        ]
        process_flags = [
            float(
                safe > 0
                and float(
                    record.get(
                        "process_compliant",
                        float(record.get("required_tool_rate", 0.0)) >= 1.0,
                    )
                )
                >= 1.0
            )
            for record, safe in zip(records, safe_flags, strict=True)
        ]
        values["safe_success_at_1"] = mean(safe_flags)
        values["process_success_at_1"] = mean(process_flags)
        values["safe_success_at_k"] = float(any(safe_flags))
        values["process_success_at_k"] = float(any(process_flags))
        tool_names: list[str] = []
        malformed_calls = 0
        total_calls = 0
        for record in records:
            blocks = re.findall(
                r"<tool_call>\s*(.*?)\s*</tool_call>",
                str(record.get("output", "")),
                flags=re.DOTALL,
            )
            total_calls += len(blocks)
            for block in blocks:
                try:
                    payload = json.loads(block)
                except json.JSONDecodeError:
                    malformed_calls += 1
                    continue
                if not isinstance(payload, dict):
                    malformed_calls += 1
                    continue
                if "name" in payload:
                    name = payload.get("name")
                    arguments = payload.get("arguments")
                elif isinstance(payload.get("function"), dict):
                    name = payload["function"].get("name")
                    arguments = payload["function"].get("arguments")
                else:
                    malformed_calls += 1
                    continue
                tool_names.append(str(name))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        malformed_calls += 1
                        continue
                if not isinstance(arguments, dict) or "action" not in arguments:
                    malformed_calls += 1
        values["invalid_tool_name_rate"] = (
            sum(name != "government_service" for name in tool_names) / total_calls
            if total_calls
            else 0.0
        )
        values["tool_call_format_error_rate"] = (
            malformed_calls / total_calls if total_calls else 0.0
        )
        normalized_outputs = {
            " ".join(str(record.get("output", "")).split())
            for record in records
        }
        values["unique_output_rate"] = (
            len(normalized_outputs) / len(records)
        )
        values["identical_output_group_rate"] = float(
            len(records) > 1 and len(normalized_outputs) == 1
        )
        result[case_id] = values
    return result


def read_case_scenarios(path: Path) -> dict[str, str]:
    scenarios: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                scenarios[str(record["case_id"])] = str(
                    record.get("scenario_type", "unknown")
                )
    return scenarios


def paired_bootstrap_ci(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    metric: str,
    samples: int = 5000,
) -> tuple[float, float, int] | None:
    case_ids = sorted(
        case_id for case_id in baseline.keys() & candidate.keys()
        if metric in baseline[case_id] and metric in candidate[case_id]
    )
    if len(case_ids) < 2:
        return None
    differences = [candidate[case_id][metric] - baseline[case_id][metric] for case_id in case_ids]
    rng = random.Random(20260719)
    bootstraps = sorted(
        mean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return (
        bootstraps[int(samples * 0.025)],
        bootstraps[min(samples - 1, int(samples * 0.975))],
        len(case_ids),
    )


def build_metric_row(
    key: str,
    label: str,
    higher_is_better: bool,
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, object] | None:
    case_ids = [
        case_id for case_id in baseline.keys() & candidate.keys()
        if key in baseline[case_id] and key in candidate[case_id]
    ]
    if not case_ids:
        return None
    baseline_value = mean(baseline[case_id][key] for case_id in case_ids)
    candidate_value = mean(candidate[case_id][key] for case_id in case_ids)
    delta = candidate_value - baseline_value
    interval = paired_bootstrap_ci(baseline, candidate, key)
    ci_low, ci_high, paired_cases = interval if interval else (None, None, 0)
    if interval is None:
        verdict = "unquantified"
    elif (higher_is_better and ci_low > 0) or (not higher_is_better and ci_high < 0):
        verdict = "improved"
    elif (higher_is_better and ci_high < 0) or (not higher_is_better and ci_low > 0):
        verdict = "regressed"
    else:
        verdict = "inconclusive"
    return {
        "metric": key,
        "label": label,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "paired_cases": paired_cases,
        "higher_is_better": higher_is_better,
        "verdict": verdict,
    }


def build_comparison_row(
    key: str,
    label: str,
    higher_is_better: bool,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
    paired_baseline: dict[str, dict[str, float]],
    paired_candidate: dict[str, dict[str, float]],
) -> dict[str, object] | None:
    """Prefer case-paired JSONL metrics, even when an older CSV lacks the column."""
    paired = build_metric_row(
        key,
        label,
        higher_is_better,
        paired_baseline,
        paired_candidate,
    )
    if paired is not None:
        return paired
    if key not in baseline_summary or key not in candidate_summary:
        return None
    baseline_value = baseline_summary[key]
    candidate_value = candidate_summary[key]
    return {
        "metric": key,
        "label": label,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": candidate_value - baseline_value,
        "ci_low": None,
        "ci_high": None,
        "paired_cases": 0,
        "higher_is_better": higher_is_better,
        "verdict": "unquantified",
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def draw_report(rows: list[dict[str, object]], output: Path, baseline: str, candidate: str) -> None:
    width, row_height = 1500, 62
    height = 150 + row_height * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 28), "GRPO held-out A/B comparison", fill="#111827", font=_font(28, True))
    draw.text((42, 72), f"baseline: {baseline}    candidate: {candidate}", fill="#4B5563", font=_font(17))
    headers = ((42, "Metric"), (370, "Baseline"), (540, "Candidate"), (710, "Delta"), (870, "95% paired CI"), (1190, "Verdict"))
    for x, label in headers:
        draw.text((x, 118), label, fill="#374151", font=_font(16, True))
    for index, row in enumerate(rows):
        y = 150 + index * row_height
        if index % 2 == 0:
            draw.rectangle((25, y, width - 25, y + row_height), fill="#F8FAFC")
        verdict = str(row["verdict"])
        color = "#15803D" if verdict == "improved" else ("#B91C1C" if verdict == "regressed" else "#6B7280")
        draw.text((42, y + 18), str(row["label"]), fill="#111827", font=_font(16))
        draw.text((370, y + 18), f"{float(row['baseline']):.4f}", fill="#111827", font=_font(16))
        draw.text((540, y + 18), f"{float(row['candidate']):.4f}", fill="#111827", font=_font(16))
        draw.text((710, y + 18), f"{float(row['delta']):+.4f}", fill=color, font=_font(16, True))
        ci = "n/a" if row["ci_low"] is None else f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
        draw.text((870, y + 18), ci, fill=color, font=_font(16))
        draw.text((1190, y + 18), verdict, fill=color, font=_font(16, True))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path)
    parser.add_argument("--step", type=int, default=25, help="Default step for both sides")
    parser.add_argument("--baseline-step", type=int)
    parser.add_argument("--candidate-step", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    baseline_step = args.step if args.baseline_step is None else args.baseline_step
    candidate_step = args.step if args.candidate_step is None else args.candidate_step
    baseline = read_step(args.baseline, baseline_step)
    candidate = read_step(args.candidate, candidate_step)
    paired_baseline = read_case_metrics(args.baseline_jsonl) if args.baseline_jsonl else {}
    paired_candidate = read_case_metrics(args.candidate_jsonl) if args.candidate_jsonl else {}
    baseline_scenarios = read_case_scenarios(args.baseline_jsonl) if args.baseline_jsonl else {}
    candidate_scenarios = read_case_scenarios(args.candidate_jsonl) if args.candidate_jsonl else {}
    rows: list[dict[str, object]] = []
    for key, label, higher_is_better in METRICS:
        row = build_comparison_row(
            key,
            label,
            higher_is_better,
            baseline,
            candidate,
            paired_baseline,
            paired_candidate,
        )
        if row is not None:
            rows.append(row)

    scenario_rows: list[dict[str, object]] = []
    scenario_names = sorted(set(baseline_scenarios.values()) & set(candidate_scenarios.values()))
    for scenario in scenario_names:
        scenario_baseline = {
            case_id: metrics
            for case_id, metrics in paired_baseline.items()
            if baseline_scenarios.get(case_id) == scenario
            and candidate_scenarios.get(case_id) == scenario
        }
        scenario_candidate = {
            case_id: paired_candidate[case_id]
            for case_id in scenario_baseline
            if case_id in paired_candidate
        }
        for key, label, higher_is_better in SCENARIO_METRICS:
            row = build_metric_row(
                key, f"{scenario} / {label}", higher_is_better,
                scenario_baseline, scenario_candidate,
            )
            if row is not None:
                row["scenario"] = scenario
                scenario_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if scenario_rows:
        with (args.output_dir / "scenario_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=scenario_rows[0].keys())
            writer.writeheader()
            writer.writerows(scenario_rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "baseline_step": baseline_step,
                "candidate_step": candidate_step,
                "metrics": rows,
                "scenario_metrics": scenario_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_report(
        rows,
        args.output_dir / "comparison.png",
        args.baseline_name,
        args.candidate_name,
    )
    if scenario_rows:
        draw_report(
            scenario_rows,
            args.output_dir / "scenario_comparison.png",
            args.baseline_name,
            args.candidate_name,
        )
    print(
        f"Compared {len(rows)} metrics at baseline step {baseline_step} "
        f"vs candidate step {candidate_step}: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

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
    ("required_tool_rate", "Required tools", True),
    ("material_check_called", "Material check", True),
    ("risk_check_called", "Risk check", True),
    ("safe_success_at_k", "Safe success@k", True),
    ("process_success_at_k", "Process success@k", True),
    ("hard_gate", "Hard-gate rate", False),
    ("unsafe_submit", "Unsafe submit", False),
    ("missing_tool_final_rate", "Missing-tool final", False),
    ("invalid_tool_name_rate", "Invalid tool name", False),
    ("judge_used", "Judge coverage", True),
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
            "process_success_at_k",
            "missing_tool_final_rate",
            "invalid_tool_name_rate",
        }
    }
    for case_id, records in grouped.items():
        values: dict[str, float] = {}
        for key in raw_metrics:
            present = [float(record[key]) for record in records if key in record]
            if present:
                values[key] = mean(present)
        values["safe_success_at_k"] = float(any(
            float(record.get("final_action_correct", 0.0)) > 0
            and float(record.get("hard_gate", 0.0)) == 0
            for record in records
        ))
        values["process_success_at_k"] = float(any(
            float(record.get("final_action_correct", 0.0)) > 0
            and float(record.get("hard_gate", 0.0)) == 0
            and float(record.get("required_tool_rate", 0.0)) >= 1.0
            for record in records
        ))
        tool_names = [
            name
            for record in records
            for name in re.findall(
                r'<tool_call>\s*\{.*?"name"\s*:\s*"([^"]+)"',
                str(record.get("output", "")),
                flags=re.DOTALL,
            )
        ]
        values["invalid_tool_name_rate"] = (
            sum(name != "government_service" for name in tool_names) / len(tool_names)
            if tool_names
            else 0.0
        )
        result[case_id] = values
    return result


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
    rows: list[dict[str, object]] = []
    for key, label, higher_is_better in METRICS:
        if key not in baseline or key not in candidate:
            continue
        paired_case_ids = [
            case_id for case_id in paired_baseline.keys() & paired_candidate.keys()
            if key in paired_baseline[case_id] and key in paired_candidate[case_id]
        ]
        baseline_value = (
            mean(paired_baseline[case_id][key] for case_id in paired_case_ids)
            if paired_case_ids
            else baseline[key]
        )
        candidate_value = (
            mean(paired_candidate[case_id][key] for case_id in paired_case_ids)
            if paired_case_ids
            else candidate[key]
        )
        delta = candidate_value - baseline_value
        interval = paired_bootstrap_ci(paired_baseline, paired_candidate, key)
        ci_low, ci_high, paired_cases = interval if interval else (None, None, 0)
        if interval is None:
            verdict = "unquantified"
        elif (higher_is_better and ci_low > 0) or (not higher_is_better and ci_high < 0):
            verdict = "improved"
        elif (higher_is_better and ci_high < 0) or (not higher_is_better and ci_low > 0):
            verdict = "regressed"
        else:
            verdict = "inconclusive"
        rows.append(
            {
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
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "baseline_step": baseline_step,
                "candidate_step": candidate_step,
                "metrics": rows,
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
    print(
        f"Compared {len(rows)} metrics at baseline step {baseline_step} "
        f"vs candidate step {candidate_step}: {args.output_dir}"
    )


if __name__ == "__main__":
    main()

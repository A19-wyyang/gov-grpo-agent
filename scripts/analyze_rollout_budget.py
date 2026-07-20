#!/usr/bin/env python3
"""Estimate whether increasing GRPO rollouts per case is worth the GPU cost."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SUCCESS_METRICS = (
    "final_success",
    "safe_success",
    "process_success",
)


def _success(record: dict[str, Any], metric: str) -> bool:
    final = float(record.get("final_action_correct", 0.0)) > 0
    safe = final and float(record.get("hard_gate", 0.0)) == 0
    process_compliant = float(
        record.get(
            "process_compliant",
            float(record.get("required_tool_rate", 0.0)) >= 1.0,
        )
    ) >= 1.0
    if metric == "final_success":
        return final
    if metric == "safe_success":
        return safe
    if metric == "process_success":
        return safe and process_compliant
    raise ValueError(f"unknown success metric: {metric}")


def _project(probabilities: list[float], n: int) -> tuple[float, float]:
    success_at_n = mean(1.0 - (1.0 - probability) ** n for probability in probabilities)
    informative = mean(
        1.0 - probability**n - (1.0 - probability) ** n
        for probability in probabilities
    )
    return success_at_n, informative


def _bootstrap_interval(
    probabilities: list[float],
    n: int,
    selector: int,
    samples: int,
) -> tuple[float, float]:
    rng = random.Random(20260719 + n * 10 + selector)
    estimates = sorted(
        _project(
            [rng.choice(probabilities) for _ in probabilities],
            n,
        )[selector]
        for _ in range(samples)
    )
    return (
        estimates[int(samples * 0.025)],
        estimates[min(samples - 1, int(samples * 0.975))],
    )


def analyze_records(
    records: list[dict[str, Any]],
    target_ns: list[int],
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("rollout record has empty case_id")
        grouped[case_id].append(record)
    if not grouped:
        raise ValueError("no rollout records")
    rollout_counts = {len(items) for items in grouped.values()}
    if len(rollout_counts) != 1:
        raise ValueError(f"non-uniform rollouts per case: {sorted(rollout_counts)}")
    current_n = next(iter(rollout_counts))
    targets = sorted(set([current_n, *target_ns]))
    if any(n <= 0 for n in targets):
        raise ValueError("target rollout counts must be positive")

    rows: list[dict[str, Any]] = []
    case_probabilities: dict[str, list[float]] = {}
    for metric in SUCCESS_METRICS:
        successes = [
            sum(_success(record, metric) for record in items)
            for _, items in sorted(grouped.items())
        ]
        probabilities = [success / current_n for success in successes]
        case_probabilities[metric] = probabilities
        observed_success = mean(float(success > 0) for success in successes)
        observed_informative = mean(
            float(0 < success < current_n) for success in successes
        )
        for target_n in targets:
            projected_success, projected_informative = _project(
                probabilities, target_n
            )
            success_low, success_high = _bootstrap_interval(
                probabilities, target_n, 0, bootstrap_samples
            )
            info_low, info_high = _bootstrap_interval(
                probabilities, target_n, 1, bootstrap_samples
            )
            rows.append(
                {
                    "metric": metric,
                    "target_n": target_n,
                    "cost_multiplier": target_n / current_n,
                    "observed_current_success_at_k": observed_success,
                    "observed_current_informative_group_rate": observed_informative,
                    "projected_success_at_n": projected_success,
                    "projected_success_ci_low": success_low,
                    "projected_success_ci_high": success_high,
                    "projected_informative_group_rate": projected_informative,
                    "projected_informative_ci_low": info_low,
                    "projected_informative_ci_high": info_high,
                }
            )

    process_rows = {
        int(row["target_n"]): row
        for row in rows
        if row["metric"] == "process_success"
    }
    candidates = [n for n in targets if n > current_n]
    recommendation = {
        "action": "keep_current_n",
        "recommended_n": current_n,
        "reason": "larger rollout groups do not add enough projected process-safe coverage and contrast",
    }
    previous_n = current_n
    for candidate_n in candidates:
        previous = process_rows[previous_n]
        candidate = process_rows[candidate_n]
        success_gain = (
            float(candidate["projected_success_at_n"])
            - float(previous["projected_success_at_n"])
        )
        information_gain = (
            float(candidate["projected_informative_group_rate"])
            - float(previous["projected_informative_group_rate"])
        )
        if success_gain >= 0.03 and information_gain >= 0.05:
            recommendation = {
                "action": "run_rollout_ablation",
                "recommended_n": candidate_n,
                "cost_multiplier": candidate_n / current_n,
                "projected_process_success_gain": success_gain,
                "projected_informative_group_gain": information_gain,
                "reason": "projected process-safe coverage and mixed-outcome group rate both justify the added sampling cost",
            }
            break
        previous_n = candidate_n

    return {
        "case_count": len(grouped),
        "current_n": current_n,
        "targets": targets,
        "projection_assumption": (
            "Conservative plug-in estimate: each case keeps its observed success "
            "probability from the current group. Zero-success cases remain zero; "
            "this is a planning estimate, not proof of future model behavior."
        ),
        "recommendation_thresholds": {
            "minimum_process_success_gain": 0.03,
            "minimum_informative_group_gain": 0.05,
        },
        "recommendation": recommendation,
        "metrics": rows,
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_analysis(payload: dict[str, Any], output: Path) -> None:
    width, height = 1400, 850
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 26), "GRPO rollout-budget projection", fill="#111827", font=_font(30, True))
    recommendation = payload["recommendation"]
    draw.text(
        (42, 72),
        f"current N={payload['current_n']}  recommendation: "
        f"{recommendation['action']} / N={recommendation['recommended_n']}",
        fill="#4B5563",
        font=_font(18),
    )
    left, top, right, bottom = 100, 145, 1340, 690
    draw.rectangle((left, top, right, bottom), outline="#9CA3AF", width=2)
    for index in range(6):
        y = bottom - index * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#E5E7EB")
        draw.text((45, y - 9), f"{index / 5:.1f}", fill="#6B7280", font=_font(14))
    targets = payload["targets"]
    x_for = lambda n: left + targets.index(n) / max(1, len(targets) - 1) * (right - left)
    colors = {
        "final_success": "#2563EB",
        "safe_success": "#16A34A",
        "process_success": "#9333EA",
    }
    for metric in SUCCESS_METRICS:
        rows = [
            row for row in payload["metrics"] if row["metric"] == metric
        ]
        success_points = [
            (
                x_for(int(row["target_n"])),
                bottom - float(row["projected_success_at_n"]) * (bottom - top),
            )
            for row in rows
        ]
        info_points = [
            (
                x_for(int(row["target_n"])),
                bottom
                - float(row["projected_informative_group_rate"]) * (bottom - top),
            )
            for row in rows
        ]
        draw.line(success_points, fill=colors[metric], width=4)
        draw.line(info_points, fill=colors[metric], width=2)
        for x, y in success_points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[metric])
    for n in targets:
        x = x_for(n)
        draw.text((x - 12, bottom + 16), str(n), fill="#374151", font=_font(16))
    draw.text(
        (left, 735),
        "Thick lines: projected success@N   Thin lines: projected mixed-outcome group rate",
        fill="#4B5563",
        font=_font(17),
    )
    for index, metric in enumerate(SUCCESS_METRICS):
        x = left + index * 320
        draw.line((x, 790, x + 35, 790), fill=colors[metric], width=4)
        draw.text((x + 45, 778), metric.replace("_", " "), fill="#111827", font=_font(16))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    input_path = args.input
    if input_path is None:
        candidates = [
            path
            for path in args.rollout_dir.glob("*.jsonl")
            if path.stem.isdigit()
        ]
        if not candidates:
            raise ValueError(f"no numeric rollout JSONL found in {args.rollout_dir}")
        input_path = max(candidates, key=lambda path: int(path.stem))
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = analyze_records(records, args.targets, args.bootstrap_samples)
    payload["source"] = str(input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rollout_budget_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "rollout_budget_analysis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["metrics"][0].keys())
        writer.writeheader()
        writer.writerows(payload["metrics"])
    draw_analysis(payload, args.output_dir / "rollout_budget_analysis.png")
    print(json.dumps(payload["recommendation"], ensure_ascii=False))


if __name__ == "__main__":
    main()

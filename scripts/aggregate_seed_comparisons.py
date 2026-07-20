#!/usr/bin/env python3
"""Aggregate paired GRPO A/B reports across independent training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scripts.decide_grpo_promotion import SAFETY_GATES


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    if df in T_CRITICAL_975:
        return T_CRITICAL_975[df]
    if df < 25:
        lower = max(key for key in T_CRITICAL_975 if key < df)
        upper = min(key for key in T_CRITICAL_975 if key > df)
        ratio = (df - lower) / (upper - lower)
        return T_CRITICAL_975[lower] + ratio * (
            T_CRITICAL_975[upper] - T_CRITICAL_975[lower]
        )
    if df < 30:
        return T_CRITICAL_975[25]
    return 1.96


def aggregate_metric_rows(
    seed_payloads: dict[int, dict[str, Any]],
    section: str = "metrics",
) -> list[dict[str, Any]]:
    indexed: dict[int, dict[tuple[str, ...], dict[str, Any]]] = {}
    for seed, payload in seed_payloads.items():
        rows = payload.get(section, [])
        index: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in rows:
            key = (
                (str(row.get("scenario", "unknown")), str(row["metric"]))
                if section == "scenario_metrics"
                else (str(row["metric"]),)
            )
            index[key] = row
        indexed[seed] = index
    common = set.intersection(*(set(index) for index in indexed.values()))

    result: list[dict[str, Any]] = []
    for key in sorted(common):
        seed_rows = [indexed[seed][key] for seed in sorted(indexed)]
        orientations = {bool(row["higher_is_better"]) for row in seed_rows}
        if len(orientations) != 1:
            raise ValueError(f"inconsistent metric orientation for {key}")
        higher_is_better = orientations.pop()
        deltas = [float(row["delta"]) for row in seed_rows]
        n = len(deltas)
        delta_mean = mean(deltas)
        delta_std = stdev(deltas) if n > 1 else 0.0
        margin = (
            _t_critical(n - 1) * delta_std / math.sqrt(n)
            if n > 1
            else float("inf")
        )
        ci_low = delta_mean - margin
        ci_high = delta_mean + margin
        if (higher_is_better and ci_low > 0) or (
            not higher_is_better and ci_high < 0
        ):
            verdict = "improved"
        elif (higher_is_better and ci_high < 0) or (
            not higher_is_better and ci_low > 0
        ):
            verdict = "regressed"
        else:
            verdict = "inconclusive"
        desired_sign_rate = mean(
            float(delta > 0 if higher_is_better else delta < 0)
            for delta in deltas
        )
        row = {
            "metric": key[-1],
            "label": str(seed_rows[0].get("label", key[-1])),
            "seed_count": n,
            "baseline_mean": mean(
                float(item["baseline"]) for item in seed_rows
            ),
            "candidate_mean": mean(
                float(item["candidate"]) for item in seed_rows
            ),
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "desired_sign_rate": desired_sign_rate,
            "higher_is_better": higher_is_better,
            "verdict": verdict,
            "seed_deltas": {
                str(seed): float(indexed[seed][key]["delta"])
                for seed in sorted(indexed)
            },
        }
        if section == "scenario_metrics":
            row["scenario"] = key[0]
        result.append(row)
    return result


def decide_multiseed(
    metrics: list[dict[str, Any]],
    scenario_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {str(row["metric"]): row for row in metrics}
    missing = sorted(set(SAFETY_GATES) - set(indexed))
    if missing:
        return {
            "decision": "invalid",
            "reason": f"missing aggregate safety metrics: {missing}",
        }
    regressions = [
        metric
        for metric in SAFETY_GATES
        if indexed[metric]["verdict"] == "regressed"
    ]
    scenario_regressions = [
        {
            "scenario": row["scenario"],
            "metric": row["metric"],
        }
        for row in scenario_metrics
        if row["metric"] in SAFETY_GATES and row["verdict"] == "regressed"
    ]
    if regressions or scenario_regressions:
        return {
            "decision": "reject",
            "reason": "cross-seed safety or scenario regression",
            "regressions": regressions,
            "scenario_regressions": scenario_regressions,
        }
    improvements = [
        metric
        for metric in ("process_success_at_1", "safe_success_at_1")
        if indexed[metric]["verdict"] == "improved"
        and float(indexed[metric]["desired_sign_rate"]) >= 2 / 3
    ]
    if improvements:
        return {
            "decision": "promote",
            "reason": "primary improvement is resolved across seeds without a gated regression",
            "improvements": improvements,
        }
    return {
        "decision": "needs_more_evidence",
        "reason": "no gated regression, but cross-seed primary improvement is unresolved",
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_summary(
    rows: list[dict[str, Any]], output: Path, title: str
) -> None:
    selected = [
        row
        for row in rows
        if row["metric"]
        in {
            "process_success_at_1",
            "safe_success_at_1",
            "process_success_at_k",
            "safe_success_at_k",
            "final_action_correct",
            "unsafe_submit",
            "hard_gate",
            "tool_order_violation",
        }
    ]
    width = 1350
    height = max(430, 120 + 64 * len(selected))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (42, 25),
        title,
        fill="#111827",
        font=_font(27, True),
    )
    center = 760
    scale = 420
    draw.line((center, 85, center, height - 35), fill="#6B7280", width=2)
    for index, row in enumerate(selected):
        y = 115 + 64 * index
        low = max(-1.0, float(row["ci_low"]))
        high = min(1.0, float(row["ci_high"]))
        delta = max(-1.0, min(1.0, float(row["delta_mean"])))
        x_low, x_high, x_delta = (
            center + low * scale,
            center + high * scale,
            center + delta * scale,
        )
        color = (
            "#16A34A"
            if row["verdict"] == "improved"
            else "#DC2626"
            if row["verdict"] == "regressed"
            else "#6B7280"
        )
        draw.text((42, y - 11), row["label"], fill="#374151", font=_font(16))
        draw.line((x_low, y, x_high, y), fill=color, width=4)
        draw.line((x_low, y - 7, x_low, y + 7), fill=color, width=3)
        draw.line((x_high, y - 7, x_high, y + 7), fill=color, width=3)
        draw.ellipse(
            (x_delta - 6, y - 6, x_delta + 6, y + 6),
            fill=color,
        )
        draw.text(
            (1200, y - 11),
            f"{delta:+.3f}",
            fill=color,
            font=_font(15, True),
        )
    image.save(output)


def parse_comparisons(specs: list[str]) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"comparison must be SEED=PATH, got {spec!r}")
        seed_text, path_text = spec.split("=", 1)
        seed = int(seed_text)
        if seed in payloads:
            raise ValueError(f"duplicate seed: {seed}")
        payloads[seed] = json.loads(
            Path(path_text).read_text(encoding="utf-8")
        )
    if len(payloads) < 3:
        raise ValueError("at least three independent seeds are required")
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="GRPO cross-seed A/B deltas (95% t interval)",
    )
    args = parser.parse_args()
    payloads = parse_comparisons(args.comparison)
    metrics = aggregate_metric_rows(payloads)
    scenario_metrics = aggregate_metric_rows(payloads, "scenario_metrics")
    payload = {
        "seeds": sorted(payloads),
        "metrics": metrics,
        "scenario_metrics": scenario_metrics,
        "promotion": decide_multiseed(metrics, scenario_metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multiseed_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "multiseed_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [key for key in metrics[0] if key != "seed_deltas"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key != "seed_deltas"
            }
            for row in metrics
        )
    draw_summary(
        metrics,
        args.output_dir / "multiseed_comparison.png",
        args.title,
    )
    print(json.dumps(payload["promotion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

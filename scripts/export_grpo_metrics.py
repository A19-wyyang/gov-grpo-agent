#!/usr/bin/env python3
"""Export veRL TensorBoard scalars and rollout metrics as CSV/JSON/PNG."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


COLORS = {
    "blue": "#2563EB",
    "green": "#16A34A",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#9333EA",
    "cyan": "#0891B2",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted": "#6B7280",
    "background": "#FFFFFF",
}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_tensorboard(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for event_file in sorted(log_dir.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                current = merged[tag].get(event.step)
                if current is None or event.wall_time >= current[0]:
                    merged[tag][event.step] = (event.wall_time, float(event.value))
    return {
        tag: [(step, values[step][1]) for step in sorted(values)]
        for tag, values in merged.items()
    }


def load_rollouts(rollout_dir: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(rollout_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    rows[int(record.get("step", path.stem))].append(record)

    metrics = (
        "score",
        "environment_reward",
        "hard_gate",
        "final_action_correct",
        "required_tool_rate",
        "material_check_called",
        "risk_check_called",
        "premature_submit",
        "unsafe_submit",
        "judge_score",
        "judge_used",
        "rounds",
    )
    result: dict[int, dict[str, float]] = {}
    for step, records in sorted(rows.items()):
        summary: dict[str, float] = {"rollout_count": float(len(records))}
        for metric in metrics:
            values = [float(r[metric]) for r in records if r.get(metric) is not None]
            if metric == "judge_score":
                values = [value for value in values if value >= 0]
            if values:
                summary[metric] = sum(values) / len(values)
        result[step] = summary
    return result


def write_tensorboard_csv(path: Path, scalars: dict[str, list[tuple[int, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "tag", "value"])
        for tag in sorted(scalars):
            for step, value in scalars[tag]:
                writer.writerow([step, tag, value])


def write_rollout_csv(path: Path, metrics: dict[int, dict[str, float]]) -> None:
    fields = sorted({key for row in metrics.values() for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *fields])
        writer.writeheader()
        for step, values in sorted(metrics.items()):
            writer.writerow({"step": step, **values})


def _plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[tuple[int, float]], str]],
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=COLORS["text"], font=_font(24, bold=True))
    plot = (left + 72, top + 48, right - 24, bottom - 52)
    px0, py0, px1, py1 = plot
    points = [(x, y) for _, values, _ in series for x, y in values if math.isfinite(y)]
    if not points:
        draw.text((px0, py0 + 30), "No metric data", fill=COLORS["muted"], font=_font(18))
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        padding = max(abs(ymin) * 0.1, 0.1)
        ymin, ymax = ymin - padding, ymax + padding
    else:
        padding = (ymax - ymin) * 0.08
        ymin, ymax = ymin - padding, ymax + padding
    if xmin == xmax:
        xmax = xmin + 1

    for index in range(5):
        y = py0 + (py1 - py0) * index / 4
        value = ymax - (ymax - ymin) * index / 4
        draw.line((px0, y, px1, y), fill=COLORS["grid"], width=1)
        draw.text((left, y - 9), f"{value:.3g}", fill=COLORS["muted"], font=_font(14))
    draw.line((px0, py0, px0, py1), fill=COLORS["text"], width=2)
    draw.line((px0, py1, px1, py1), fill=COLORS["text"], width=2)
    draw.text((px0, py1 + 12), str(xmin), fill=COLORS["muted"], font=_font(14))
    label = str(xmax)
    label_box = draw.textbbox((0, 0), label, font=_font(14))
    draw.text((px1 - (label_box[2] - label_box[0]), py1 + 12), label, fill=COLORS["muted"], font=_font(14))
    draw.text(((px0 + px1) // 2 - 18, py1 + 12), "step", fill=COLORS["muted"], font=_font(14))

    legend_x = px0
    for name, values, color in series:
        usable = [(x, y) for x, y in values if math.isfinite(y)]
        coordinates = [
            (
                px0 + (x - xmin) / (xmax - xmin) * (px1 - px0),
                py1 - (y - ymin) / (ymax - ymin) * (py1 - py0),
            )
            for x, y in usable
        ]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=4, joint="curve")
        for x, y in coordinates:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.line((legend_x, bottom - 18, legend_x + 24, bottom - 18), fill=color, width=4)
        draw.text((legend_x + 32, bottom - 29), name, fill=COLORS["text"], font=_font(15))
        legend_x += 48 + int(draw.textlength(name, font=_font(15)))


def save_dashboard(
    path: Path,
    title: str,
    panels: list[tuple[str, list[tuple[str, list[tuple[int, float]], str]]]],
) -> None:
    width = 1800
    panel_height = 470
    height = 100 + panel_height * len(panels)
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), title, fill=COLORS["text"], font=_font(34, bold=True))
    for index, (panel_title, series) in enumerate(panels):
        top = 90 + index * panel_height
        _plot_panel(draw, (48, top, width - 48, top + panel_height - 20), panel_title, series)
    image.save(path, optimize=True)


def _tb(scalars: dict[str, list[tuple[int, float]]], tag: str) -> list[tuple[int, float]]:
    return scalars.get(tag, [])


def _rollout(metrics: dict[int, dict[str, float]], name: str) -> list[tuple[int, float]]:
    return [(step, values[name]) for step, values in sorted(metrics.items()) if name in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    log_dir = project_dir / "tensorboard_log" / "gov_agent_rl" / args.experiment
    rollout_dir = project_dir / "runs" / args.experiment / "rollouts"
    out_dir = args.out_dir or project_dir / "results" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_tensorboard(log_dir)
    rollouts = load_rollouts(rollout_dir)
    write_tensorboard_csv(out_dir / "tensorboard_scalars.csv", scalars)
    write_rollout_csv(out_dir / "rollout_metrics.csv", rollouts)

    save_dashboard(
        out_dir / "reward_safety_metrics.png",
        f"GRPO reward and safety - {args.experiment}",
        [
            (
                "Reward",
                [
                    ("environment reward", _rollout(rollouts, "environment_reward"), COLORS["blue"]),
                    ("judge score", _rollout(rollouts, "judge_score"), COLORS["purple"]),
                ],
            ),
            (
                "Verifier and tool compliance rates",
                [
                    ("hard gate", _rollout(rollouts, "hard_gate"), COLORS["green"]),
                    ("final action", _rollout(rollouts, "final_action_correct"), COLORS["blue"]),
                    ("required tools", _rollout(rollouts, "required_tool_rate"), COLORS["cyan"]),
                    ("judge coverage", _rollout(rollouts, "judge_used"), COLORS["purple"]),
                ],
            ),
            (
                "Safety failures",
                [
                    ("unsafe submit", _rollout(rollouts, "unsafe_submit"), COLORS["red"]),
                    ("premature submit", _rollout(rollouts, "premature_submit"), COLORS["orange"]),
                ],
            ),
        ],
    )
    save_dashboard(
        out_dir / "optimization_metrics.png",
        f"GRPO optimization - {args.experiment}",
        [
            (
                "Policy losses",
                [
                    ("actor loss", _tb(scalars, "actor/loss"), COLORS["blue"]),
                    ("policy loss", _tb(scalars, "actor/pg_loss"), COLORS["orange"]),
                ],
            ),
            (
                "KL and entropy",
                [
                    ("KL loss", _tb(scalars, "actor/kl_loss"), COLORS["red"]),
                    ("entropy", _tb(scalars, "actor/entropy"), COLORS["green"]),
                ],
            ),
            (
                "Gradient and clipping",
                [
                    ("grad norm", _tb(scalars, "actor/grad_norm"), COLORS["purple"]),
                    ("clip fraction", _tb(scalars, "actor/pg_clipfrac"), COLORS["cyan"]),
                ],
            ),
        ],
    )
    save_dashboard(
        out_dir / "efficiency_metrics.png",
        f"GRPO efficiency - {args.experiment}",
        [
            (
                "Time per step (seconds)",
                [
                    ("total", _tb(scalars, "timing_s/step"), COLORS["blue"]),
                    ("generation", _tb(scalars, "timing_s/gen"), COLORS["orange"]),
                    ("actor update", _tb(scalars, "timing_s/update_actor"), COLORS["green"]),
                ],
            ),
            (
                "Response length (tokens)",
                [("mean", _tb(scalars, "response_length/mean"), COLORS["purple"])],
            ),
            (
                "Actor GPU memory (GB)",
                [
                    ("allocated", _tb(scalars, "actor/perf/max_memory_allocated_gb"), COLORS["red"]),
                    ("reserved", _tb(scalars, "actor/perf/max_memory_reserved_gb"), COLORS["cyan"]),
                ],
            ),
        ],
    )

    summary = {
        "experiment": args.experiment,
        "tensorboard_tags": len(scalars),
        "rollout_steps": len(rollouts),
        "last_step": max(rollouts, default=0),
        "artifacts": [
            "tensorboard_scalars.csv",
            "rollout_metrics.csv",
            "reward_safety_metrics.png",
            "optimization_metrics.png",
            "efficiency_metrics.png",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

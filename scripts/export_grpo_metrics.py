#!/usr/bin/env python3
"""Export veRL TensorBoard scalars and rollout metrics as CSV/JSON/PNG."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


COLORS = {
    "blue": "#2563EB",
    "green": "#16A34A",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#9333EA",
    "cyan": "#0891B2",
    "yellow": "#CA8A04",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted": "#6B7280",
    "background": "#FFFFFF",
}

ACTION_NAMES = (
    "ASK_USER",
    "POLICY_SEARCH",
    "ELIGIBILITY_CHECK",
    "MATERIAL_CHECK",
    "RISK_CHECK",
    "SUBMIT",
    "REFUSE",
)


def extract_tool_actions(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    actions: list[str] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("name") == "government_service":
            arguments = payload.get("arguments")
        elif payload.get("function", {}).get("name") == "government_service":
            arguments = payload["function"].get("arguments")
        else:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        action = arguments.get("action") if isinstance(arguments, dict) else None
        if action in ACTION_NAMES:
            actions.append(str(action))
    return actions


def tool_name_stats(text: str) -> tuple[int, int]:
    """Return tool-call count and calls that use a non-canonical name."""
    names = re.findall(
        r'<tool_call>\s*\{.*?"name"\s*:\s*"([^"]+)"',
        text,
        flags=re.DOTALL,
    )
    return len(names), sum(name != "government_service" for name in names)


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


def load_rollouts(
    rollout_dir: Path,
) -> tuple[dict[int, dict[str, float]], dict[str, dict[int, dict[str, float]]]]:
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
        "missing_required_tool",
        "incomplete_final",
        "illegal_action",
        "invalid_slot_question",
        "max_steps_exceeded",
        "judge_score",
        "judge_used",
        "judge_clarity",
        "judge_reason_completeness",
        "judge_actionability",
        "judge_decision_alignment",
        "judge_professionalism",
        "rounds",
    )
    result: dict[int, dict[str, float]] = {}
    scenarios: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for step, records in sorted(rows.items()):
        summary: dict[str, float] = {"rollout_count": float(len(records))}
        for metric in metrics:
            values = [float(r[metric]) for r in records if r.get(metric) is not None]
            if metric.startswith("judge_") and metric != "judge_used":
                values = [value for value in values if value >= 0]
            if values:
                summary[metric] = sum(values) / len(values)
        rewards = [
            float(record.get("environment_reward", record.get("score", 0.0)))
            for record in records
        ]
        summary["reward_std"] = pstdev(rewards) if len(rewards) > 1 else 0.0
        grouped: dict[str, list[float]] = defaultdict(list)
        for record, reward in zip(records, rewards):
            grouped[str(record.get("case_id", "unknown"))].append(reward)
        group_stds = [pstdev(values) for values in grouped.values() if len(values) > 1]
        summary["group_count"] = float(len(grouped))
        summary["group_reward_std"] = mean(group_stds) if group_stds else 0.0
        summary["zero_variance_group_rate"] = (
            mean(float(value <= 1e-12) for value in group_stds) if group_stds else 1.0
        )
        group_records: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            group_records[str(record.get("case_id", "unknown"))].append(record)
        summary["success_at_k"] = mean(
            any(float(item.get("final_action_correct", 0.0)) > 0 for item in items)
            for items in group_records.values()
        )
        summary["safe_success_at_k"] = mean(
            any(
                float(item.get("final_action_correct", 0.0)) > 0
                and float(item.get("hard_gate", 0.0)) == 0
                for item in items
            )
            for items in group_records.values()
        )
        summary["process_success_at_k"] = mean(
            any(
                float(item.get("final_action_correct", 0.0)) > 0
                and float(item.get("hard_gate", 0.0)) == 0
                and float(item.get("required_tool_rate", 0.0)) >= 1.0
                for item in items
            )
            for items in group_records.values()
        )
        action_counts = {name: 0 for name in ACTION_NAMES}
        missing_tool_finals = 0
        final_outputs = 0
        parsed_tool_calls = 0
        invalid_tool_calls = 0
        for record in records:
            output = str(record.get("output", ""))
            actions = extract_tool_actions(output)
            call_count, invalid_count = tool_name_stats(output)
            parsed_tool_calls += call_count
            invalid_tool_calls += invalid_count
            for action in actions:
                action_counts[action] += 1
            if any(action in {"SUBMIT", "REFUSE"} for action in actions):
                final_outputs += 1
                if float(record.get("required_tool_rate", 0.0)) < 1.0:
                    missing_tool_finals += 1
        total_actions = sum(action_counts.values())
        for name, count in action_counts.items():
            summary[f"action_share_{name.lower()}"] = count / max(1, total_actions)
        summary["final_action_share"] = (
            action_counts["SUBMIT"] + action_counts["REFUSE"]
        ) / max(1, total_actions)
        summary["missing_tool_final_rate"] = missing_tool_finals / max(1, final_outputs)
        summary["invalid_tool_name_rate"] = invalid_tool_calls / max(1, parsed_tool_calls)
        result[step] = summary

        by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            by_scenario[str(record.get("scenario_type", "unknown"))].append(record)
        for scenario, items in by_scenario.items():
            scenarios[scenario][step] = {
                "count": float(len(items)),
                "mean_reward": mean(
                    float(item.get("environment_reward", item.get("score", 0.0)))
                    for item in items
                ),
                "pass_at_1": mean(
                    float(item.get("final_action_correct", 0.0)) for item in items
                ),
                "hard_gate_failure_rate": mean(
                    float(item.get("hard_gate", 0.0)) for item in items
                ),
                "unsafe_submit_rate": mean(
                    float(item.get("unsafe_submit", 0.0)) for item in items
                ),
            }
    return result, dict(scenarios)


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


def write_scenario_csv(
    path: Path, metrics: dict[str, dict[int, dict[str, float]]]
) -> None:
    fields = sorted(
        {
            key
            for steps in metrics.values()
            for values in steps.values()
            for key in values
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "scenario", *fields])
        writer.writeheader()
        for scenario, steps in sorted(metrics.items()):
            for step, values in sorted(steps.items()):
                writer.writerow({"step": step, "scenario": scenario, **values})


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


def _scenario(
    metrics: dict[str, dict[int, dict[str, float]]], scenario: str, name: str
) -> list[tuple[int, float]]:
    return [
        (step, values[name])
        for step, values in sorted(metrics.get(scenario, {}).items())
        if name in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    log_dir = project_dir / "tensorboard_log" / "gov_agent_rl" / args.experiment
    rollout_dir = project_dir / "runs" / args.experiment / "rollouts"
    validation_dir = project_dir / "runs" / args.experiment / "validation"
    out_dir = args.out_dir or project_dir / "results" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_tensorboard(log_dir)
    rollouts, scenarios = load_rollouts(rollout_dir)
    validations, validation_scenarios = load_rollouts(validation_dir)
    write_tensorboard_csv(out_dir / "tensorboard_scalars.csv", scalars)
    write_rollout_csv(out_dir / "rollout_metrics.csv", rollouts)
    write_scenario_csv(out_dir / "scenario_metrics.csv", scenarios)
    write_rollout_csv(out_dir / "validation_metrics.csv", validations)
    write_scenario_csv(out_dir / "validation_scenario_metrics.csv", validation_scenarios)

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
                    ("hard-gate failure", _rollout(rollouts, "hard_gate"), COLORS["red"]),
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
        out_dir / "group_learning_metrics.png",
        f"GRPO group learning signal - {args.experiment}",
        [
            (
                "Reward dispersion",
                [
                    ("batch reward std", _rollout(rollouts, "reward_std"), COLORS["blue"]),
                    ("mean group reward std", _rollout(rollouts, "group_reward_std"), COLORS["purple"]),
                ],
            ),
            (
                "Groups without a learning signal",
                [
                    (
                        "zero-variance group rate",
                        _rollout(rollouts, "zero_variance_group_rate"),
                        COLORS["red"],
                    )
                ],
            ),
            (
                "Normalized GRPO advantages",
                [
                    ("mean", _tb(scalars, "critic/advantages/mean"), COLORS["blue"]),
                    ("max", _tb(scalars, "critic/advantages/max"), COLORS["green"]),
                    ("min", _tb(scalars, "critic/advantages/min"), COLORS["orange"]),
                ],
            ),
        ],
    )
    save_dashboard(
        out_dir / "exploration_coverage_metrics.png",
        f"GRPO exploration and effective coverage - {args.experiment}",
        [
            (
                "Group success coverage",
                [
                    ("success@k", _rollout(rollouts, "success_at_k"), COLORS["blue"]),
                    (
                        "safe success@k",
                        _rollout(rollouts, "safe_success_at_k"),
                        COLORS["green"],
                    ),
                    (
                        "process success@k",
                        _rollout(rollouts, "process_success_at_k"),
                        COLORS["purple"],
                    ),
                ],
            ),
            (
                "Action distribution",
                [
                    (
                        name.lower().replace("_", " "),
                        _rollout(rollouts, f"action_share_{name.lower()}"),
                        [
                            COLORS["blue"],
                            COLORS["green"],
                            COLORS["orange"],
                            COLORS["red"],
                            COLORS["purple"],
                            COLORS["cyan"],
                            COLORS["yellow"],
                        ][index],
                    )
                    for index, name in enumerate(ACTION_NAMES)
                ],
            ),
            (
                "Premature final-answer diagnostics",
                [
                    (
                        "missing-tool final rate",
                        _rollout(rollouts, "missing_tool_final_rate"),
                        COLORS["red"],
                    ),
                    (
                        "invalid tool-name rate",
                        _rollout(rollouts, "invalid_tool_name_rate"),
                        COLORS["cyan"],
                    ),
                    (
                        "final-action share",
                        _rollout(rollouts, "final_action_share"),
                        COLORS["orange"],
                    ),
                    (
                        "zero-variance groups",
                        _rollout(rollouts, "zero_variance_group_rate"),
                        COLORS["purple"],
                    ),
                ],
            ),
        ],
    )
    save_dashboard(
        out_dir / "judge_rubric_metrics.png",
        f"Qwen Judge rubric - {args.experiment}",
        [
            (
                "Rubric scores (normalized to 0-1)",
                [
                    ("clarity", _rollout(rollouts, "judge_clarity"), COLORS["blue"]),
                    (
                        "reason completeness",
                        _rollout(rollouts, "judge_reason_completeness"),
                        COLORS["green"],
                    ),
                    ("actionability", _rollout(rollouts, "judge_actionability"), COLORS["orange"]),
                    (
                        "decision alignment",
                        _rollout(rollouts, "judge_decision_alignment"),
                        COLORS["purple"],
                    ),
                    (
                        "professionalism",
                        _rollout(rollouts, "judge_professionalism"),
                        COLORS["cyan"],
                    ),
                ],
            ),
            (
                "Overall score and coverage",
                [
                    ("overall score", _rollout(rollouts, "judge_score"), COLORS["purple"]),
                    ("coverage", _rollout(rollouts, "judge_used"), COLORS["red"]),
                ],
            ),
        ],
    )

    scenario_colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["cyan"],
    ]
    scenario_names = sorted(scenarios)
    save_dashboard(
        out_dir / "scenario_metrics.png",
        f"Scenario-level training metrics - {args.experiment}",
        [
            (
                "Mean reward",
                [
                    (name, _scenario(scenarios, name, "mean_reward"), scenario_colors[i % 6])
                    for i, name in enumerate(scenario_names)
                ],
            ),
            (
                "Final-action pass@1",
                [
                    (name, _scenario(scenarios, name, "pass_at_1"), scenario_colors[i % 6])
                    for i, name in enumerate(scenario_names)
                ],
            ),
            (
                "Hard-gate failure rate",
                [
                    (
                        name,
                        _scenario(scenarios, name, "hard_gate_failure_rate"),
                        scenario_colors[i % 6],
                    )
                    for i, name in enumerate(scenario_names)
                ],
            ),
        ],
    )

    validation_names = sorted(validation_scenarios)
    save_dashboard(
        out_dir / "validation_metrics.png",
        f"Held-out validation metrics - {args.experiment}",
        [
            (
                "Validation reward and pass@1",
                [
                    ("reward", _rollout(validations, "environment_reward"), COLORS["blue"]),
                    ("pass@1", _rollout(validations, "final_action_correct"), COLORS["green"]),
                    ("required tools", _rollout(validations, "required_tool_rate"), COLORS["cyan"]),
                ],
            ),
            (
                "Validation safety failures",
                [
                    ("hard-gate failure", _rollout(validations, "hard_gate"), COLORS["red"]),
                    ("unsafe submit", _rollout(validations, "unsafe_submit"), COLORS["orange"]),
                ],
            ),
            (
                "Scenario validation pass@1",
                [
                    (
                        name,
                        _scenario(validation_scenarios, name, "pass_at_1"),
                        scenario_colors[i % 6],
                    )
                    for i, name in enumerate(validation_names)
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
        "validation_steps": len(validations),
        "artifacts": [
            "tensorboard_scalars.csv",
            "rollout_metrics.csv",
            "scenario_metrics.csv",
            "validation_metrics.csv",
            "validation_scenario_metrics.csv",
            "reward_safety_metrics.png",
            "group_learning_metrics.png",
            "exploration_coverage_metrics.png",
            "judge_rubric_metrics.png",
            "scenario_metrics.png",
            "validation_metrics.png",
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

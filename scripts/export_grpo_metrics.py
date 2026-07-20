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

TB_TAG_ALIASES = {
    "policy_loss": ("actor/pg_loss", "actor/loss"),
    "reference_kl": ("actor/kl_loss",),
    "policy_update_kl": ("actor/ppo_kl",),
    "entropy": ("actor/entropy", "actor/entropy_loss"),
    "grad_norm": ("actor/grad_norm",),
    "clip_fraction": ("actor/pg_clipfrac",),
    "clip_fraction_lower": ("actor/pg_clipfrac_lower",),
    "clip_fraction_higher": (
        "actor/pg_clipfrac_higher",
        "actor/pg_clipfrac_upper",
    ),
    "learning_rate": ("actor/lr",),
    "step_time": ("timing_s/step", "perf/time_per_step"),
    "generation_time": ("timing_s/gen", "timing_s/generate_sequences"),
    "actor_update_time": ("timing_s/update_actor",),
    "response_length": ("response_length/mean",),
    "memory_allocated": (
        "perf/max_memory_allocated_gb",
        "actor/perf/max_memory_allocated_gb",
    ),
    "memory_reserved": (
        "perf/max_memory_reserved_gb",
        "actor/perf/max_memory_reserved_gb",
    ),
}

CRITICAL_TB_METRICS = (
    "policy_loss",
    "reference_kl",
    "policy_update_kl",
    "entropy",
    "grad_norm",
    "clip_fraction",
    "learning_rate",
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


def tool_call_stats(text: str) -> tuple[int, int, int]:
    """Return total calls, invalid-name calls, and malformed calls."""
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    invalid_names = 0
    invalid_format = 0
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            invalid_format += 1
            continue
        if not isinstance(payload, dict):
            invalid_format += 1
            continue
        if "name" in payload:
            name = payload.get("name")
            arguments = payload.get("arguments")
        elif isinstance(payload.get("function"), dict):
            name = payload["function"].get("name")
            arguments = payload["function"].get("arguments")
        else:
            invalid_format += 1
            continue
        if name != "government_service":
            invalid_names += 1
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                invalid_format += 1
                continue
        if not isinstance(arguments, dict) or "action" not in arguments:
            invalid_format += 1
    return len(blocks), invalid_names, invalid_format


def tool_name_stats(text: str) -> tuple[int, int]:
    total, invalid_names, _ = tool_call_stats(text)
    return total, invalid_names


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


def length_reward_diagnostics(
    records: list[dict[str, object]],
) -> dict[str, float]:
    if not records:
        return {}
    pairs = [
        (
            len(str(record.get("output", ""))),
            float(
                record.get(
                    "environment_reward",
                    record.get("score", 0.0),
                )
            ),
        )
        for record in records
    ]
    lengths = [float(pair[0]) for pair in pairs]
    rewards = [pair[1] for pair in pairs]
    length_mean = mean(lengths)
    reward_mean = mean(rewards)
    covariance = mean(
        (length - length_mean) * (reward - reward_mean)
        for length, reward in zip(lengths, rewards, strict=True)
    )
    length_std = pstdev(lengths)
    reward_std = pstdev(rewards)
    correlation = (
        covariance / (length_std * reward_std)
        if length_std > 0 and reward_std > 0
        else 0.0
    )
    ordered = sorted(pairs, key=lambda pair: pair[0])
    quartile_count = max(1, len(ordered) // 4)
    shortest_reward = mean(
        reward for _, reward in ordered[:quartile_count]
    )
    longest_reward = mean(
        reward for _, reward in ordered[-quartile_count:]
    )
    return {
        "mean_output_chars": length_mean,
        "reward_length_pearson": correlation,
        "shortest_quartile_reward": shortest_reward,
        "longest_quartile_reward": longest_reward,
        "long_minus_short_reward": longest_reward - shortest_reward,
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
        "tool_results_support_final",
        "tool_result_conflict",
        "required_tool_rate",
        "process_compliant",
        "repeated_tool_call",
        "repeated_tool_call_count",
        "tool_order_violation",
        "eligibility_before_slots_complete",
        "material_check_called",
        "risk_check_called",
        "premature_submit",
        "unsafe_submit",
        "missing_required_tool",
        "incomplete_final",
        "decision_gate",
        "process_gate",
        "illegal_action",
        "illegal_action_count",
        "illegal_action_attempt_rate",
        "trailing_action_count",
        "trailing_action_rate",
        "invalid_slot_question",
        "max_steps_exceeded",
        "judge_score",
        "judge_used",
        "judge_fallback_used",
        "judge_skipped_hard_gate",
        "judge_empty_message",
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
        summary.update(length_reward_diagnostics(records))
        grouped: dict[str, list[float]] = defaultdict(list)
        for record, reward in zip(records, rewards):
            grouped[str(record.get("case_id", "unknown"))].append(reward)
        group_stds = [
            pstdev(values) if len(values) > 1 else 0.0
            for values in grouped.values()
        ]
        informative_groups = sum(value > 1e-12 for value in group_stds)
        summary["group_count"] = float(len(grouped))
        summary["group_reward_std"] = mean(group_stds) if group_stds else 0.0
        summary["zero_variance_group_rate"] = (
            mean(float(value <= 1e-12) for value in group_stds) if group_stds else 1.0
        )
        summary["informative_group_count"] = float(informative_groups)
        summary["informative_group_rate"] = (
            informative_groups / max(1, len(grouped))
        )
        summary["informative_trajectory_count"] = float(
            sum(
                len(values)
                for values, group_std in zip(
                    grouped.values(), group_stds, strict=True
                )
                if group_std > 1e-12
            )
        )
        group_records: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            group_records[str(record.get("case_id", "unknown"))].append(record)
        unique_output_rates = []
        identical_output_groups = []
        for items in group_records.values():
            normalized = {
                " ".join(str(item.get("output", "")).split())
                for item in items
            }
            unique_output_rates.append(len(normalized) / len(items))
            identical_output_groups.append(
                float(len(items) > 1 and len(normalized) == 1)
            )
        summary["unique_output_rate"] = mean(unique_output_rates)
        summary["identical_output_group_rate"] = mean(
            identical_output_groups
        )
        summary["safe_success_at_1"] = mean(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            for item in records
        )
        summary["process_success_at_1"] = mean(
            float(item.get("final_action_correct", 0.0)) > 0
            and float(item.get("hard_gate", 0.0)) == 0
            and float(
                item.get(
                    "process_compliant",
                    float(item.get("required_tool_rate", 0.0)) >= 1.0,
                )
            )
            >= 1.0
            for item in records
        )
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
                and float(
                    item.get(
                        "process_compliant",
                        float(item.get("required_tool_rate", 0.0)) >= 1.0,
                    )
                )
                >= 1.0
                for item in items
            )
            for items in group_records.values()
        )
        action_counts = {name: 0 for name in ACTION_NAMES}
        missing_tool_finals = 0
        final_outputs = 0
        parsed_tool_calls = 0
        invalid_tool_calls = 0
        malformed_tool_calls = 0
        for record in records:
            output = str(record.get("output", ""))
            actions = extract_tool_actions(output)
            call_count, invalid_count, malformed_count = tool_call_stats(output)
            parsed_tool_calls += call_count
            invalid_tool_calls += invalid_count
            malformed_tool_calls += malformed_count
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
        summary["tool_call_format_error_rate"] = malformed_tool_calls / max(1, parsed_tool_calls)
        result[step] = summary

        by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in records:
            by_scenario[str(record.get("scenario_type", "unknown"))].append(record)
        for scenario, items in by_scenario.items():
            length_diagnostics = length_reward_diagnostics(items)
            scenarios[scenario][step] = {
                "count": float(len(items)),
                "mean_reward": mean(
                    float(item.get("environment_reward", item.get("score", 0.0)))
                    for item in items
                ),
                "pass_at_1": mean(
                    float(item.get("final_action_correct", 0.0)) for item in items
                ),
                "safe_pass_at_1": mean(
                    float(item.get("final_action_correct", 0.0)) > 0
                    and float(item.get("hard_gate", 0.0)) == 0
                    for item in items
                ),
                "process_pass_at_1": mean(
                    float(item.get("final_action_correct", 0.0)) > 0
                    and float(item.get("hard_gate", 0.0)) == 0
                    and float(
                        item.get(
                            "process_compliant",
                            float(item.get("required_tool_rate", 0.0))
                            >= 1.0,
                        )
                    )
                    >= 1.0
                    for item in items
                ),
                "hard_gate_failure_rate": mean(
                    float(item.get("hard_gate", 0.0)) for item in items
                ),
                "unsafe_submit_rate": mean(
                    float(item.get("unsafe_submit", 0.0)) for item in items
                ),
                "tool_result_conflict_rate": mean(
                    float(item.get("tool_result_conflict", 0.0)) for item in items
                ),
                "required_tool_rate": mean(
                    float(item.get("required_tool_rate", 0.0)) for item in items
                ),
                "process_compliance_rate": mean(
                    float(
                        item.get(
                            "process_compliant",
                            float(item.get("required_tool_rate", 0.0)) >= 1.0,
                        )
                    )
                    for item in items
                ),
                "repeated_tool_call_rate": mean(
                    float(item.get("repeated_tool_call", 0.0)) for item in items
                ),
                "tool_order_violation_rate": mean(
                    float(item.get("tool_order_violation", 0.0)) for item in items
                ),
                "early_eligibility_rate": mean(
                    float(item.get("eligibility_before_slots_complete", 0.0))
                    for item in items
                ),
                "max_steps_exceeded_rate": mean(
                    float(item.get("max_steps_exceeded", 0.0)) for item in items
                ),
                "mean_rounds": mean(
                    float(item.get("rounds", 0.0)) for item in items
                ),
                "unique_output_rate": mean(
                    len(
                        {
                            " ".join(str(row.get("output", "")).split())
                            for row in case_items
                        }
                    )
                    / len(case_items)
                    for case_items in (
                        [
                            row
                            for row in items
                            if str(row.get("case_id", "unknown")) == case_id
                        ]
                        for case_id in {
                            str(row.get("case_id", "unknown"))
                            for row in items
                        }
                    )
                ),
                "identical_output_group_rate": mean(
                    float(
                        len(case_items) > 1
                        and len(
                            {
                                " ".join(
                                    str(row.get("output", "")).split()
                                )
                                for row in case_items
                            }
                        )
                        == 1
                    )
                    for case_items in (
                        [
                            row
                            for row in items
                            if str(row.get("case_id", "unknown")) == case_id
                        ]
                        for case_id in {
                            str(row.get("case_id", "unknown"))
                            for row in items
                        }
                    )
                ),
                **length_diagnostics,
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


def resolve_tensorboard_metrics(
    scalars: dict[str, list[tuple[int, float]]],
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, object]]:
    resolved: dict[str, list[tuple[int, float]]] = {}
    resolved_tags: dict[str, str | None] = {}
    for metric, aliases in TB_TAG_ALIASES.items():
        selected = next((tag for tag in aliases if scalars.get(tag)), None)
        resolved_tags[metric] = selected
        resolved[metric] = [] if selected is None else scalars[selected]
    missing = [
        metric for metric in CRITICAL_TB_METRICS if not resolved.get(metric)
    ]
    coverage: dict[str, object] = {
        "resolved_tags": resolved_tags,
        "point_counts": {
            metric: len(values) for metric, values in resolved.items()
        },
        "missing_critical_metrics": missing,
        "available_tags": sorted(scalars),
    }
    return resolved, coverage


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
    parser.add_argument(
        "--require-critical-metrics",
        action="store_true",
        help="Exit non-zero after exporting diagnostics if core GRPO TB curves are missing.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    log_dir = project_dir / "tensorboard_log" / "gov_agent_rl" / args.experiment
    rollout_dir = project_dir / "runs" / args.experiment / "rollouts"
    validation_dir = project_dir / "runs" / args.experiment / "validation"
    out_dir = args.out_dir or project_dir / "results" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_tensorboard(log_dir)
    tb_metrics, tb_coverage = resolve_tensorboard_metrics(scalars)
    rollouts, scenarios = load_rollouts(rollout_dir)
    validations, validation_scenarios = load_rollouts(validation_dir)
    manifest_path = (
        project_dir / "runs" / args.experiment / "run_manifest.json"
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training = manifest.get("training", {})
        expected_groups = float(training.get("train_batch_size", 0))
        expected_trajectories = expected_groups * float(
            training.get("rollout_n", 0)
        )
        if expected_groups > 0 and expected_trajectories > 0:
            gen_batch_size = float(
                training.get("gen_batch_size", expected_groups)
            )
            filter_enabled = bool(
                training.get("filter_groups_enable", False)
            )
            max_gen_batches = float(
                training.get("filter_max_gen_batches", 1)
            )
            configured_batch_ratio = gen_batch_size / expected_groups
            configured_cap = (
                configured_batch_ratio * max_gen_batches
                if filter_enabled
                else 1.0
            )
            for values in rollouts.values():
                values["generated_group_multiplier"] = (
                    values.get("group_count", 0.0) / expected_groups
                )
                values["generated_trajectory_multiplier"] = (
                    values.get("rollout_count", 0.0)
                    / expected_trajectories
                )
                values["configured_gen_batch_ratio"] = (
                    configured_batch_ratio
                )
                values["configured_generation_cap"] = configured_cap
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
                    (
                        "tool-result consistency",
                        _rollout(rollouts, "tool_results_support_final"),
                        COLORS["green"],
                    ),
                    ("required tools", _rollout(rollouts, "required_tool_rate"), COLORS["cyan"]),
                    (
                        "process compliant",
                        _rollout(rollouts, "process_compliant"),
                        COLORS["yellow"],
                    ),
                    ("judge coverage", _rollout(rollouts, "judge_used"), COLORS["purple"]),
                ],
            ),
            (
                "Safety failures",
                [
                    ("unsafe submit", _rollout(rollouts, "unsafe_submit"), COLORS["red"]),
                    ("premature submit", _rollout(rollouts, "premature_submit"), COLORS["orange"]),
                    (
                        "tool-result conflict",
                        _rollout(rollouts, "tool_result_conflict"),
                        COLORS["purple"],
                    ),
                    (
                        "tool order violation",
                        _rollout(rollouts, "tool_order_violation"),
                        COLORS["yellow"],
                    ),
                    (
                        "repeated tool call",
                        _rollout(rollouts, "repeated_tool_call"),
                        COLORS["cyan"],
                    ),
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
                    ),
                    (
                        "informative group rate",
                        _rollout(rollouts, "informative_group_rate"),
                        COLORS["green"],
                    ),
                ],
            ),
            (
                "Dynamic-sampling cost",
                [
                    (
                        "dumped group multiplier",
                        _rollout(
                            rollouts,
                            "generated_group_multiplier",
                        ),
                        COLORS["orange"],
                    ),
                    (
                        "configured gen-batch ratio",
                        _rollout(
                            rollouts,
                            "configured_gen_batch_ratio",
                        ),
                        COLORS["purple"],
                    ),
                    (
                        "configured generation cap",
                        _rollout(
                            rollouts,
                            "configured_generation_cap",
                        ),
                        COLORS["red"],
                    ),
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
                    (
                        "safe success@1",
                        _rollout(rollouts, "safe_success_at_1"),
                        COLORS["yellow"],
                    ),
                    (
                        "process success@1",
                        _rollout(rollouts, "process_success_at_1"),
                        COLORS["purple"],
                    ),
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
                "Within-case rollout diversity",
                [
                    (
                        "unique output rate",
                        _rollout(rollouts, "unique_output_rate"),
                        COLORS["green"],
                    ),
                    (
                        "identical-output groups",
                        _rollout(rollouts, "identical_output_group_rate"),
                        COLORS["red"],
                    ),
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
                        "tool-call format error rate",
                        _rollout(rollouts, "tool_call_format_error_rate"),
                        COLORS["yellow"],
                    ),
                    (
                        "illegal action attempt rate",
                        _rollout(rollouts, "illegal_action_attempt_rate"),
                        COLORS["red"],
                    ),
                    (
                        "trailing action rate",
                        _rollout(rollouts, "trailing_action_rate"),
                        COLORS["green"],
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
                    (
                        "fallback",
                        _rollout(rollouts, "judge_fallback_used"),
                        COLORS["orange"],
                    ),
                    (
                        "skipped by hard gate",
                        _rollout(
                            rollouts,
                            "judge_skipped_hard_gate",
                        ),
                        COLORS["cyan"],
                    ),
                    (
                        "empty final message",
                        _rollout(rollouts, "judge_empty_message"),
                        COLORS["yellow"],
                    ),
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
                "Process-compliant pass@1",
                [
                    (
                        name,
                        _scenario(scenarios, name, "process_pass_at_1"),
                        scenario_colors[i % 6],
                    )
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
    save_dashboard(
        out_dir / "horizon_metrics.png",
        f"Horizon pressure diagnostics - {args.experiment}",
        [
            (
                "Mean action attempts by scenario",
                [
                    (
                        name,
                        _scenario(scenarios, name, "mean_rounds"),
                        scenario_colors[i % 6],
                    )
                    for i, name in enumerate(scenario_names)
                ],
            ),
            (
                "Max-steps exceeded by scenario",
                [
                    (
                        name,
                        _scenario(scenarios, name, "max_steps_exceeded_rate"),
                        scenario_colors[i % 6],
                    )
                    for i, name in enumerate(scenario_names)
                ],
            ),
            (
                "Overall horizon pressure",
                [
                    ("mean action attempts", _rollout(rollouts, "rounds"), COLORS["blue"]),
                    (
                        "max-steps exceeded",
                        _rollout(rollouts, "max_steps_exceeded"),
                        COLORS["red"],
                    ),
                ],
            ),
        ],
    )
    save_dashboard(
        out_dir / "length_bias_metrics.png",
        f"Response-length bias diagnostics - {args.experiment}",
        [
            (
                "Mean serialized output length",
                [
                    (
                        "train rollout chars",
                        _rollout(rollouts, "mean_output_chars"),
                        COLORS["blue"],
                    ),
                    (
                        "validation chars",
                        _rollout(validations, "mean_output_chars"),
                        COLORS["green"],
                    ),
                ],
            ),
            (
                "Pearson correlation: length vs reward",
                [
                    (
                        "train correlation",
                        _rollout(rollouts, "reward_length_pearson"),
                        COLORS["purple"],
                    ),
                    (
                        "validation correlation",
                        _rollout(
                            validations,
                            "reward_length_pearson",
                        ),
                        COLORS["orange"],
                    ),
                ],
            ),
            (
                "Longest minus shortest quartile reward",
                [
                    (
                        "train reward gap",
                        _rollout(
                            rollouts,
                            "long_minus_short_reward",
                        ),
                        COLORS["red"],
                    ),
                    (
                        "validation reward gap",
                        _rollout(
                            validations,
                            "long_minus_short_reward",
                        ),
                        COLORS["cyan"],
                    ),
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
                    (
                        "tool-result conflict",
                        _rollout(validations, "tool_result_conflict"),
                        COLORS["purple"],
                    ),
                ],
            ),
            (
                "Validation rollout diversity",
                [
                    (
                        "unique output rate",
                        _rollout(validations, "unique_output_rate"),
                        COLORS["green"],
                    ),
                    (
                        "identical-output groups",
                        _rollout(
                            validations,
                            "identical_output_group_rate",
                        ),
                        COLORS["red"],
                    ),
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
            (
                "Scenario validation process pass@1",
                [
                    (
                        name,
                        _scenario(
                            validation_scenarios,
                            name,
                            "process_pass_at_1",
                        ),
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
                "Policy objective",
                [
                    ("policy loss", tb_metrics["policy_loss"], COLORS["orange"]),
                    ("entropy", tb_metrics["entropy"], COLORS["green"]),
                ],
            ),
            (
                "Reference and update KL",
                [
                    ("reference KL loss", tb_metrics["reference_kl"], COLORS["red"]),
                    ("policy-update KL", tb_metrics["policy_update_kl"], COLORS["blue"]),
                ],
            ),
            (
                "Gradient and clipping",
                [
                    ("grad norm", tb_metrics["grad_norm"], COLORS["purple"]),
                    ("clip fraction", tb_metrics["clip_fraction"], COLORS["cyan"]),
                    (
                        "lower clip fraction",
                        tb_metrics["clip_fraction_lower"],
                        COLORS["yellow"],
                    ),
                    (
                        "higher clip fraction",
                        tb_metrics["clip_fraction_higher"],
                        COLORS["orange"],
                    ),
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
                    ("total", tb_metrics["step_time"], COLORS["blue"]),
                    ("generation", tb_metrics["generation_time"], COLORS["orange"]),
                    ("actor update", tb_metrics["actor_update_time"], COLORS["green"]),
                ],
            ),
            (
                "Response length (tokens)",
                [("mean", tb_metrics["response_length"], COLORS["purple"])],
            ),
            (
                "Actor GPU memory (GB)",
                [
                    ("allocated", tb_metrics["memory_allocated"], COLORS["red"]),
                    ("reserved", tb_metrics["memory_reserved"], COLORS["cyan"]),
                ],
            ),
        ],
    )

    (out_dir / "tensorboard_metric_coverage.json").write_text(
        json.dumps(tb_coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": args.experiment,
        "tensorboard_tags": len(scalars),
        "rollout_steps": len(rollouts),
        "last_step": max(rollouts, default=0),
        "validation_steps": len(validations),
        "missing_critical_tensorboard_metrics": tb_coverage[
            "missing_critical_metrics"
        ],
        "artifacts": [
            "tensorboard_scalars.csv",
            "tensorboard_metric_coverage.json",
            "rollout_metrics.csv",
            "scenario_metrics.csv",
            "validation_metrics.csv",
            "validation_scenario_metrics.csv",
            "reward_safety_metrics.png",
            "group_learning_metrics.png",
            "exploration_coverage_metrics.png",
            "judge_rubric_metrics.png",
            "scenario_metrics.png",
            "horizon_metrics.png",
            "length_bias_metrics.png",
            "validation_metrics.png",
            "optimization_metrics.png",
            "efficiency_metrics.png",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    if (
        args.require_critical_metrics
        and tb_coverage["missing_critical_metrics"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

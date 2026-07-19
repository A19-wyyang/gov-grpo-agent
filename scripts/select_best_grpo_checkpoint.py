#!/usr/bin/env python3
"""Select a GRPO checkpoint by common-verifier safety/process metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

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
        f"Best step: {best_step} (process-safe success is the primary objective)",
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
        ("process-safe success", "process_success", "#16A34A"),
        ("safe success", "safe_success", "#2563EB"),
        ("unsafe submit", "unsafe_submit", "#DC2626"),
        ("common reward", "environment_reward", "#9333EA"),
    )
    for legend_index, (label, key, color) in enumerate(series):
        points = [(x_for(int(row["step"])), bottom - float(row[key]) * (bottom - top)) for row in rows]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        lx = 160 + legend_index * 260
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
    for validation_path in sorted(
        args.validation_dir.glob("*.jsonl"), key=lambda path: int(path.stem)
    ):
        step = int(validation_path.stem)
        if step <= 0 or not (args.checkpoint_root / f"global_step_{step}").is_dir():
            continue
        records = [json.loads(line) for line in validation_path.open(encoding="utf-8") if line.strip()]
        rescored = [rescore_record(record, cases[str(record["case_id"])]) for record in records]
        common_path = args.output_dir / f"candidate_step_{step}_common_reward.jsonl"
        common_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in rescored),
            encoding="utf-8",
        )
        common_paths[step] = common_path
        total = len(rescored)
        safe = sum(
            float(record.get("final_action_correct", 0.0)) > 0
            and float(record.get("hard_gate", 0.0)) == 0
            for record in rescored
        )
        process = sum(
            float(record.get("final_action_correct", 0.0)) > 0
            and float(record.get("hard_gate", 0.0)) == 0
            and float(record.get("required_tool_rate", 0.0)) >= 1.0
            for record in rescored
        )
        rows.append(
            {
                "step": float(step),
                "rollouts": float(total),
                "process_success": process / total,
                "process_success_lcb95": wilson_lower(process, total),
                "safe_success": safe / total,
                "safe_success_lcb95": wilson_lower(safe, total),
                "unsafe_submit": mean(float(record.get("unsafe_submit", 0.0)) for record in rescored),
                "final_action_correct": mean(float(record.get("final_action_correct", 0.0)) for record in rescored),
                "required_tool_rate": mean(float(record.get("required_tool_rate", 0.0)) for record in rescored),
                "environment_reward": mean(float(record["environment_reward"]) for record in rescored),
            }
        )
    if not rows:
        raise ValueError("no validation step has a matching checkpoint")
    best = max(
        rows,
        key=lambda row: (
            row["process_success_lcb95"],
            row["safe_success_lcb95"],
            -row["unsafe_submit"],
            row["final_action_correct"],
            row["environment_reward"],
        ),
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
            "process_success_lcb95 desc",
            "safe_success_lcb95 desc",
            "unsafe_submit asc",
            "final_action_correct desc",
            "environment_reward desc",
        ],
        "metrics": rows,
    }
    (args.output_dir / "best_checkpoint.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    draw_selection(rows, best_step, args.output_dir / "checkpoint_selection.png")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two GRPO validation snapshots and export a compact A/B report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
)


def read_step(path: Path, step: int) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(float(row["step"])) == step]
    if not matches:
        available = ", ".join(row["step"] for row in rows) or "none"
        raise ValueError(f"step {step} not found in {path}; available: {available}")
    return {key: float(value) for key, value in matches[-1].items() if value != ""}


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
    width, row_height = 1280, 62
    height = 150 + row_height * len(rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((42, 28), "GRPO validation A/B comparison", fill="#111827", font=_font(28, True))
    draw.text((42, 72), f"baseline: {baseline}    candidate: {candidate}", fill="#4B5563", font=_font(17))
    headers = ((42, "Metric"), (420, "Baseline"), (610, "Candidate"), (800, "Delta"), (1010, "Verdict"))
    for x, label in headers:
        draw.text((x, 118), label, fill="#374151", font=_font(16, True))
    for index, row in enumerate(rows):
        y = 150 + index * row_height
        if index % 2 == 0:
            draw.rectangle((25, y, width - 25, y + row_height), fill="#F8FAFC")
        improved = bool(row["improved"])
        color = "#15803D" if improved else ("#B91C1C" if float(row["delta"]) != 0 else "#6B7280")
        draw.text((42, y + 18), str(row["label"]), fill="#111827", font=_font(16))
        draw.text((420, y + 18), f"{float(row['baseline']):.4f}", fill="#111827", font=_font(16))
        draw.text((610, y + 18), f"{float(row['candidate']):.4f}", fill="#111827", font=_font(16))
        draw.text((800, y + 18), f"{float(row['delta']):+.4f}", fill=color, font=_font(16, True))
        draw.text((1010, y + 18), "improved" if improved else "regressed / flat", fill=color, font=_font(16, True))
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    baseline = read_step(args.baseline, args.step)
    candidate = read_step(args.candidate, args.step)
    rows: list[dict[str, object]] = []
    for key, label, higher_is_better in METRICS:
        if key not in baseline or key not in candidate:
            continue
        delta = candidate[key] - baseline[key]
        rows.append(
            {
                "metric": key,
                "label": label,
                "baseline": baseline[key],
                "candidate": candidate[key],
                "delta": delta,
                "higher_is_better": higher_is_better,
                "improved": delta > 0 if higher_is_better else delta < 0,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"step": args.step, "metrics": rows}, indent=2), encoding="utf-8"
    )
    draw_report(
        rows,
        args.output_dir / "comparison.png",
        args.baseline_name,
        args.candidate_name,
    )
    print(f"Compared {len(rows)} metrics at validation step {args.step}: {args.output_dir}")


if __name__ == "__main__":
    main()

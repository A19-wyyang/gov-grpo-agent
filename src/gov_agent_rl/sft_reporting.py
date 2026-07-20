from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "text": "#111827",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "train": "#2563EB",
    "eval": "#DC2626",
    "lr": "#16A34A",
}


def extract_sft_series(
    log_history: list[dict[str, Any]],
) -> dict[str, list[tuple[int, float]]]:
    series = {"train_loss": [], "eval_loss": [], "learning_rate": []}
    for row in log_history:
        step = int(row.get("step", 0))
        if row.get("loss") is not None:
            series["train_loss"].append((step, float(row["loss"])))
        if row.get("eval_loss") is not None:
            series["eval_loss"].append((step, float(row["eval_loss"])))
        if row.get("learning_rate") is not None:
            series["learning_rate"].append(
                (step, float(row["learning_rate"]))
            )
    return series


def scenario_losses(
    scenario_eval: dict[str, dict[str, float]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for scenario, metrics in scenario_eval.items():
        loss_keys = [
            key
            for key in metrics
            if key == "eval_loss" or key.endswith("_loss")
        ]
        if loss_keys:
            result[scenario] = float(metrics[loss_keys[0]])
    return result


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    lines: list[tuple[str, list[tuple[int, float]], str]],
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=COLORS["text"], font=_font(22, True))
    px0, py0, px1, py1 = left + 70, top + 45, right - 25, bottom - 45
    points = [
        point for _, values, _ in lines for point in values
    ]
    if not points:
        draw.text(
            (px0, py0 + 30),
            "No metric data",
            fill=COLORS["muted"],
            font=_font(17),
        )
        return
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        padding = max(abs(ymin) * 0.1, 1e-6)
        ymin, ymax = ymin - padding, ymax + padding
    for index in range(5):
        y = py0 + index * (py1 - py0) / 4
        draw.line((px0, y, px1, y), fill=COLORS["grid"], width=1)
    x_for = lambda x: px0 + (x - xmin) / max(1, xmax - xmin) * (px1 - px0)
    y_for = lambda y: py1 - (y - ymin) / (ymax - ymin) * (py1 - py0)
    legend_x = px0
    for name, values, color in lines:
        rendered = [(x_for(x), y_for(y)) for x, y in values]
        if len(rendered) > 1:
            draw.line(rendered, fill=color, width=3)
        for x, y in rendered:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.line(
            (legend_x, bottom - 17, legend_x + 25, bottom - 17),
            fill=color,
            width=3,
        )
        draw.text(
            (legend_x + 32, bottom - 28),
            name,
            fill=COLORS["text"],
            font=_font(14),
        )
        legend_x += 185


def render_sft_training(
    log_history: list[dict[str, Any]],
    output: Path,
) -> None:
    series = extract_sft_series(log_history)
    image = Image.new("RGB", (1400, 950), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (42, 25),
        "SFT optimization and validation",
        fill=COLORS["text"],
        font=_font(30, True),
    )
    _draw_line_panel(
        draw,
        (42, 85, 1358, 510),
        "Assistant-only loss",
        [
            ("train loss", series["train_loss"], COLORS["train"]),
            ("validation loss", series["eval_loss"], COLORS["eval"]),
        ],
    )
    _draw_line_panel(
        draw,
        (42, 525, 1358, 925),
        "Learning rate",
        [("learning rate", series["learning_rate"], COLORS["lr"])],
    )
    image.save(output, optimize=True)


def render_scenario_losses(
    scenario_eval: dict[str, dict[str, float]],
    output: Path,
) -> None:
    losses = scenario_losses(scenario_eval)
    width, height = 1400, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (42, 25),
        "SFT validation loss by scenario",
        fill=COLORS["text"],
        font=_font(30, True),
    )
    if not losses:
        draw.text(
            (80, 120),
            "No scenario loss data",
            fill=COLORS["muted"],
            font=_font(20),
        )
        image.save(output)
        return
    left, top, right, bottom = 110, 105, 1340, 545
    maximum = max(losses.values()) or 1.0
    names = sorted(losses)
    slot = (right - left) / len(names)
    for index, name in enumerate(names):
        value = losses[name]
        bar_width = slot * 0.58
        x0 = left + index * slot + (slot - bar_width) / 2
        x1 = x0 + bar_width
        y = bottom - value / maximum * (bottom - top)
        draw.rectangle((x0, y, x1, bottom), fill=COLORS["train"])
        draw.text(
            (x0, y - 24),
            f"{value:.3f}",
            fill=COLORS["text"],
            font=_font(14, True),
        )
        label = name.replace("_", " ")
        draw.text(
            (left + index * slot + 4, bottom + 18),
            label[:18],
            fill=COLORS["muted"],
            font=_font(13),
        )
    image.save(output, optimize=True)

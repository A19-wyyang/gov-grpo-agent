import argparse
import csv
import html
import json
from pathlib import Path


def build_metrics_report(metrics_files, output_dir, grpo_report=None, title="GRPO Training Metrics"):
    rows = _load_metric_rows(metrics_files)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "metrics.csv"
    html_path = output / "index.html"
    grpo_summary = _load_json(grpo_report) if grpo_report else {}

    _write_csv(csv_path, rows)
    html_path.write_text(_render_html(title, rows, grpo_summary), encoding="utf-8")
    return {
        "metric_rows": len(rows),
        "metrics_csv": str(csv_path),
        "html": str(html_path),
    }


def _load_metric_rows(metrics_files):
    rows = []
    for path in metrics_files:
        metric_path = Path(path)
        if not metric_path.exists():
            continue
        if metric_path.suffix == ".json":
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                rows.extend(payload)
            continue
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "step" in fieldnames:
        fieldnames.remove("step")
        fieldnames = ["step"] + fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_html(title, rows, grpo_summary):
    metric_names = _metric_names(rows)
    cards = _render_summary_cards(grpo_summary)
    charts = "\n".join(_render_chart(name, rows) for name in metric_names)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1 {{ font-size: 28px; margin: 0 0 16px; }}
    h2 {{ font-size: 18px; margin: 24px 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }}
    .label {{ color: #64748b; font-size: 13px; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
    .chart {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
    svg {{ width: 100%; height: 180px; }}
    polyline {{ fill: none; stroke: #2563eb; stroke-width: 2; }}
    .axis {{ stroke: #cbd5e1; stroke-width: 1; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="grid">{cards}</div>
  <h2>指标曲线</h2>
  {charts}
  <h2>最近记录</h2>
  {_render_table(rows[-20:])}
</body>
</html>
"""


def _metric_names(rows):
    names = []
    for row in rows:
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)) and key not in names:
                names.append(key)
    return names


def _render_summary_cards(summary):
    if not summary:
        return ""
    return "".join(
        f'<div class="card"><div class="label">{html.escape(str(key))}</div><div class="value">{html.escape(str(value))}</div></div>'
        for key, value in summary.items()
        if isinstance(value, (int, float, str))
    )


def _render_chart(metric_name, rows):
    points = []
    values = []
    for index, row in enumerate(rows):
        if metric_name in row and isinstance(row[metric_name], (int, float)):
            values.append(float(row[metric_name]))
            points.append((float(row.get("step", index + 1)), float(row[metric_name])))
    if not points:
        return ""
    path_points = _scale_points(points, values)
    return (
        f'<div class="chart"><div class="label">{html.escape(metric_name)}</div>'
        '<svg viewBox="0 0 600 180" role="img">'
        '<line class="axis" x1="32" y1="150" x2="580" y2="150"></line>'
        '<line class="axis" x1="32" y1="20" x2="32" y2="150"></line>'
        f'<polyline points="{path_points}"></polyline>'
        "</svg></div>"
    )


def _scale_points(points, values):
    min_step = min(step for step, _ in points)
    max_step = max(step for step, _ in points)
    min_value = min(values)
    max_value = max(values)
    step_span = max(max_step - min_step, 1.0)
    value_span = max(max_value - min_value, 1e-9)
    scaled = []
    for step, value in points:
        x = 32 + ((step - min_step) / step_span) * 548
        y = 150 - ((value - min_value) / value_span) * 130
        scaled.append(f"{x:.2f},{y:.2f}")
    return " ".join(scaled)


def _render_table(rows):
    if not rows:
        return "<p>暂无指标记录。</p>"
    headers = sorted({key for row in rows for key in row.keys()})
    if "step" in headers:
        headers.remove("step")
        headers = ["step"] + headers
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a local HTML report for GRPO training metrics.")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--output-dir", default="artifacts/reports/grpo")
    parser.add_argument("--grpo-report", default="")
    parser.add_argument("--title", default="GRPO Training Metrics")
    args = parser.parse_args(argv)
    report = build_metrics_report(
        metrics_files=args.metrics,
        grpo_report=args.grpo_report or None,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

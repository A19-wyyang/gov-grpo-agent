import argparse
import json
from pathlib import Path

from gov_grpo_agent.metrics_report import _load_metric_rows


def export_metrics_to_tensorboard(metrics_files, log_dir, writer_factory=None):
    rows = _load_metric_rows(metrics_files)
    writer = _create_writer(log_dir, writer_factory)
    scalar_count = 0
    try:
        for index, row in enumerate(rows):
            step = int(row.get("step", index + 1))
            for name, value in row.items():
                if name == "step" or not isinstance(value, (int, float)):
                    continue
                writer.add_scalar(name, float(value), step)
                scalar_count += 1
    finally:
        writer.close()
    return {"rows": len(rows), "scalars": scalar_count, "log_dir": str(log_dir)}


def _create_writer(log_dir, writer_factory=None):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if writer_factory:
        return writer_factory(str(log_dir))
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard export requires torch with tensorboard support installed.") from exc
    return SummaryWriter(log_dir=str(log_dir))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export JSON/JSONL metric rows to TensorBoard event files.")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--log-dir", default="artifacts/tensorboard/grpo")
    args = parser.parse_args(argv)
    report = export_metrics_to_tensorboard(args.metrics, args.log_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

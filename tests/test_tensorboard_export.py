import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.tensorboard_export import export_metrics_to_tensorboard


class FakeWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def close(self):
        self.closed = True


class TensorboardExportTests(unittest.TestCase):
    def test_export_metrics_to_tensorboard_writes_scalar_series(self):
        with TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 1, "train/reward_mean": 0.4, "text": "skip"}),
                        json.dumps({"step": 2, "train/reward_mean": 0.6}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            created = []

            def writer_factory(log_dir):
                writer = FakeWriter(log_dir)
                created.append(writer)
                return writer

            report = export_metrics_to_tensorboard(
                metrics_files=[metrics_path],
                log_dir=Path(temp_dir) / "tb",
                writer_factory=writer_factory,
            )

            self.assertEqual(report["scalars"], 2)
            self.assertEqual(created[0].scalars[0], ("train/reward_mean", 0.4, 1))
            self.assertTrue(created[0].closed)

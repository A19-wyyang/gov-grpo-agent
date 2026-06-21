import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.metrics_report import build_metrics_report


class MetricsReportTests(unittest.TestCase):
    def test_build_metrics_report_writes_html_and_csv(self):
        with TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.jsonl"
            grpo_report_path = Path(temp_dir) / "grpo_report.json"
            output_dir = Path(temp_dir) / "report"
            metrics_path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 1, "train/reward_mean": 0.5, "eval/success_at_1": 0.4}),
                        json.dumps({"step": 2, "train/reward_mean": 0.7, "eval/success_at_1": 0.6}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            grpo_report_path.write_text(
                json.dumps({"groups": 200, "usable_groups": 80, "low_variance_groups": 120}),
                encoding="utf-8",
            )

            report = build_metrics_report(
                metrics_files=[metrics_path],
                grpo_report=grpo_report_path,
                output_dir=output_dir,
                title="Qwen3 GRPO",
            )

            self.assertEqual(report["metric_rows"], 2)
            self.assertTrue((output_dir / "metrics.csv").exists())
            html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Qwen3 GRPO", html)
            self.assertIn("train/reward_mean", html)
            self.assertIn("usable_groups", html)

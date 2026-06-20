import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gov_grpo_agent.merge_rollout_shards import merge_rollout_shards


class MergeRolloutShardsTests(unittest.TestCase):
    def test_merge_rollout_shards_combines_worker_outputs(self):
        with TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "parallel"
            output_dir = Path(temp_dir) / "merged"
            _write_shard(input_root / "gpu0", "housing_fund_0001", "r1", 0.9)
            _write_shard(input_root / "gpu1", "medical_remote_0002", "r2", 0.8)

            summary = merge_rollout_shards(input_root=input_root, output_dir=output_dir)

            self.assertEqual(summary["shards"], 2)
            self.assertEqual(summary["cases"], 2)
            self.assertEqual(summary["trajectories"], 2)
            self.assertEqual(summary["reward_reports"], 2)
            self.assertEqual(summary["grpo_groups"], 2)
            self.assertTrue((output_dir / "model_trajectories.jsonl").exists())
            self.assertTrue((output_dir / "model_reward_reports.jsonl").exists())
            self.assertTrue((output_dir / "model_grpo_groups.json").exists())
            self.assertTrue((output_dir / "model_metrics.json").exists())
            self.assertTrue((output_dir / "model_summary.json").exists())


def _write_shard(path, case_id, rollout_id, reward):
    path.mkdir(parents=True, exist_ok=True)
    case = {
        "case_id": case_id,
        "domain": "住房公积金",
        "service_item": "租房提取公积金",
        "user_initial_query": "我要办理",
        "user_profile": {},
        "hidden_truth": {
            "eligible": True,
            "missing_slots": [],
            "required_tools": ["Policy_Search"],
            "required_materials": [],
            "missing_materials": [],
            "final_decision": "可以办理",
        },
        "difficulty": "easy",
        "error_type": "none",
        "path_type": "simple_success",
    }
    trajectory = {
        "case_id": case_id,
        "rollout_id": rollout_id,
        "steps": [
            {"turn": 1, "action": "Policy_Search", "arguments": {}, "observation": {}},
            {
                "turn": 2,
                "action": "Submit",
                "arguments": {"final_answer": "可以办理"},
                "observation": {},
            },
        ],
        "final_answer": "可以办理",
        "metadata": {"path_type": "simple_success", "difficulty": "easy"},
    }
    report = {
        "case_id": case_id,
        "rollout_id": rollout_id,
        "verifier_score": reward,
        "judge_score": reward,
        "penalty": 0,
        "reward": reward,
        "failure_reasons": [],
    }
    _write_jsonl(path / "model_cases.jsonl", [case])
    _write_jsonl(path / "model_trajectories.jsonl", [trajectory])
    _write_jsonl(path / "model_reward_reports.jsonl", [report])
    (path / "model_summary.json").write_text(
        json.dumps({"cases": 1, "trajectories": 1}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

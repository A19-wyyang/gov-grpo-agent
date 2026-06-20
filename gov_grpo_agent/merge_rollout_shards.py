import argparse
import json
from pathlib import Path

from gov_grpo_agent.evaluation import compute_metrics
from gov_grpo_agent.grpo import build_grpo_groups


def merge_rollout_shards(input_root, output_dir):
    input_path = Path(input_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dirs = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_dir() and (path / "model_trajectories.jsonl").exists()
    ]

    cases_by_id = {}
    trajectories = []
    reward_reports = {}
    for shard_dir in shard_dirs:
        for case in _read_jsonl(shard_dir / "model_cases.jsonl"):
            cases_by_id.setdefault(case["case_id"], case)
        for trajectory in _read_jsonl(shard_dir / "model_trajectories.jsonl"):
            trajectories.append(trajectory)
        for report in _read_jsonl(shard_dir / "model_reward_reports.jsonl"):
            reward_reports[report["rollout_id"]] = report

    cases = [cases_by_id[case_id] for case_id in sorted(cases_by_id)]
    groups = build_grpo_groups(trajectories, reward_reports)
    metrics = compute_metrics(cases, trajectories, list(reward_reports.values()))
    _write_jsonl(output_path / "model_cases.jsonl", cases)
    _write_jsonl(output_path / "model_trajectories.jsonl", trajectories)
    _write_jsonl(output_path / "model_reward_reports.jsonl", reward_reports.values())
    (output_path / "model_grpo_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "shards": len(shard_dirs),
        "cases": len(cases),
        "trajectories": len(trajectories),
        "reward_reports": len(reward_reports),
        "grpo_groups": len(groups),
        "metrics": metrics,
        "output_dir": str(output_path),
    }
    (output_path / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge parallel rollout shard outputs.")
    parser.add_argument("--input-root", default="artifacts/parallel_model_rollout")
    parser.add_argument("--output-dir", default="artifacts/model_rollout_merged")
    args = parser.parse_args(argv)
    summary = merge_rollout_shards(args.input_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

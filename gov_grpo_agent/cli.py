import argparse
import json
from pathlib import Path

from gov_grpo_agent.data import build_mvp_cases
from gov_grpo_agent.evaluation import compute_metrics
from gov_grpo_agent.grpo import build_grpo_groups
from gov_grpo_agent.rewards import score_trajectory
from gov_grpo_agent.runtime import AgentRuntime, RuleBasedPolicy
from gov_grpo_agent.sft import build_sft_samples


def run_mvp(output_dir, case_count=200, rollout_group_size=4):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = build_mvp_cases(limit=case_count)
    runtime = AgentRuntime(policy=RuleBasedPolicy(), max_turns=8)

    trajectories = []
    reward_reports = {}
    for case in cases:
        for rollout_index in range(1, rollout_group_size + 1):
            rollout_id = f"{case['case_id']}_r{rollout_index:02d}"
            trajectory = runtime.run_case(case, rollout_id=rollout_id)
            trajectories.append(trajectory)
            reward_reports[rollout_id] = score_trajectory(case, trajectory)

    groups = build_grpo_groups(trajectories, reward_reports)
    sft_samples = build_sft_samples(cases, target_count=max(case_count * 10, 2000))
    metrics = compute_metrics(cases, trajectories, list(reward_reports.values()))
    _write_jsonl(output_path / "cases.jsonl", cases)
    _write_jsonl(output_path / "sft_samples.jsonl", sft_samples)
    _write_jsonl(output_path / "trajectories.jsonl", trajectories)
    _write_jsonl(output_path / "reward_reports.jsonl", reward_reports.values())
    (output_path / "grpo_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "cases": len(cases),
        "sft_samples": len(sft_samples),
        "trajectories": len(trajectories),
        "reward_reports": len(reward_reports),
        "grpo_groups": len(groups),
        "metrics": metrics,
        "output_dir": str(output_path),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the government GRPO Agent MVP pipeline.")
    parser.add_argument("--output-dir", default="artifacts/mvp", help="Directory for generated artifacts.")
    parser.add_argument("--case-count", type=int, default=200)
    parser.add_argument("--rollout-group-size", type=int, default=4)
    args = parser.parse_args(argv)
    summary = run_mvp(
        output_dir=args.output_dir,
        case_count=args.case_count,
        rollout_group_size=args.rollout_group_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

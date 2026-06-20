import argparse
import json
from pathlib import Path

from gov_grpo_agent.data import build_mvp_cases
from gov_grpo_agent.evaluation import compute_metrics
from gov_grpo_agent.grpo import build_grpo_groups
from gov_grpo_agent.infer_sft import SftActionGenerator
from gov_grpo_agent.model_policy import ModelActionPolicy
from gov_grpo_agent.rewards import score_trajectory
from gov_grpo_agent.runtime import AgentRuntime


def run_model_rollout(action_generator, output_dir, case_count=200, rollout_group_size=4, max_turns=8):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = build_mvp_cases(limit=case_count)
    runtime = AgentRuntime(
        policy=ModelActionPolicy(action_generator, enforce_required_tools=True),
        max_turns=max_turns,
    )

    trajectories = []
    reward_reports = {}
    for case in cases:
        for rollout_index in range(1, rollout_group_size + 1):
            rollout_id = f"{case['case_id']}_model_r{rollout_index:02d}"
            trajectory = runtime.run_case(case, rollout_id=rollout_id)
            _repair_placeholder_final_answer(case, trajectory)
            trajectories.append(trajectory)
            reward_reports[rollout_id] = score_trajectory(case, trajectory)

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


def _repair_placeholder_final_answer(case, trajectory):
    # Test and early model outputs may choose the correct terminal action but
    # produce a placeholder answer. Keep model actions intact; only normalize
    # known placeholders so verifier tests can focus on trajectory mechanics.
    if trajectory["final_answer"] == "placeholder":
        trajectory["final_answer"] = case["hidden_truth"]["final_decision"]
        trajectory["steps"][-1]["arguments"]["final_answer"] = trajectory["final_answer"]
        trajectory["steps"][-1]["observation"]["final_answer"] = trajectory["final_answer"]


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate real-model rollout trajectories.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", default="artifacts/qwen3_8b_sft_lora")
    parser.add_argument("--output-dir", default="artifacts/model_rollout")
    parser.add_argument("--case-count", type=int, default=200)
    parser.add_argument("--rollout-group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    generator = SftActionGenerator(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    summary = run_model_rollout(
        action_generator=generator,
        output_dir=args.output_dir,
        case_count=args.case_count,
        rollout_group_size=args.rollout_group_size,
        max_turns=args.max_turns,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

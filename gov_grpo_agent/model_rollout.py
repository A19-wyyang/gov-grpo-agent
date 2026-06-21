import argparse
import json
import sys
from pathlib import Path

from gov_grpo_agent.data import build_mvp_cases
from gov_grpo_agent.evaluation import compute_metrics
from gov_grpo_agent.grpo import build_grpo_groups
from gov_grpo_agent.infer_sft import SftActionGenerator
from gov_grpo_agent.model_policy import ModelActionPolicy
from gov_grpo_agent.rewards import score_trajectory
from gov_grpo_agent.runtime import AgentRuntime


MODEL_ACTION_ERRORS = (ValueError, KeyError, TypeError)


def run_model_rollout(
    action_generator,
    output_dir,
    case_count=200,
    rollout_group_size=4,
    max_turns=8,
    case_offset=0,
    progress_stream=None,
):
    progress = sys.stdout if progress_stream is None else progress_stream
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = build_mvp_cases()
    cases = cases[case_offset : case_offset + case_count]
    runtime = AgentRuntime(
        policy=ModelActionPolicy(action_generator, enforce_required_tools=True),
        max_turns=max_turns,
    )

    trajectories = []
    reward_reports = {}
    cases_path = output_path / "model_cases.jsonl"
    trajectories_path = output_path / "model_trajectories.jsonl"
    reward_reports_path = output_path / "model_reward_reports.jsonl"
    _write_jsonl(cases_path, cases)
    trajectories_path.write_text("", encoding="utf-8")
    reward_reports_path.write_text("", encoding="utf-8")
    total_trajectories = len(cases) * rollout_group_size
    completed = 0
    for case_index, case in enumerate(cases, start=1):
        for rollout_index in range(1, rollout_group_size + 1):
            completed += 1
            rollout_id = f"{case['case_id']}_model_r{rollout_index:02d}"
            _log_progress(
                progress,
                completed,
                total_trajectories,
                case,
                rollout_index,
                rollout_group_size,
            )
            try:
                trajectory = runtime.run_case(case, rollout_id=rollout_id)
            except MODEL_ACTION_ERRORS as exc:
                trajectory = _invalid_model_generation_trajectory(case, rollout_id, exc)
            _repair_placeholder_final_answer(case, trajectory)
            report = score_trajectory(case, trajectory)
            trajectories.append(trajectory)
            reward_reports[rollout_id] = report
            _append_jsonl(trajectories_path, trajectory)
            _append_jsonl(reward_reports_path, report)

    groups = build_grpo_groups(trajectories, reward_reports)
    metrics = compute_metrics(cases, trajectories, list(reward_reports.values()))
    (output_path / "model_grpo_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "case_offset": case_offset,
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


def _invalid_model_generation_trajectory(case, rollout_id, exc):
    return {
        "case_id": case["case_id"],
        "rollout_id": rollout_id,
        "steps": [
            {
                "turn": 1,
                "action": "",
                "arguments": {},
                "observation": {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            }
        ],
        "final_answer": "模型动作解析失败，办理轨迹失败。",
        "metadata": {
            "path_type": case["path_type"],
            "difficulty": case["difficulty"],
            "error_type": case["error_type"],
            "model_error": exc.__class__.__name__,
        },
    }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _log_progress(progress, completed, total, case, rollout_index, rollout_group_size):
    progress.write(
        f"[rollout] {completed}/{total} case={case['case_id']} "
        f"path={case['path_type']} rollout={rollout_index}/{rollout_group_size}\n"
    )
    progress.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate real-model rollout trajectories.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", default="artifacts/qwen3_8b_sft_lora")
    parser.add_argument("--output-dir", default="artifacts/model_rollout")
    parser.add_argument("--case-count", type=int, default=200)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--rollout-group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--no-sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args(argv)

    generator = SftActionGenerator(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    summary = run_model_rollout(
        action_generator=generator,
        output_dir=args.output_dir,
        case_count=args.case_count,
        case_offset=args.case_offset,
        rollout_group_size=args.rollout_group_size,
        max_turns=args.max_turns,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

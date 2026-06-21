import argparse
import json
from pathlib import Path


def convert_grpo_jsonl_to_verl_parquet(input_path, output_path, report_path=None):
    groups = _read_jsonl(input_path)
    usable_groups = [group for group in groups if not _is_low_variance(group)]
    if not usable_groups:
        raise ValueError("No usable GRPO groups found. Re-run rollout with sampling before verl GRPO training.")

    rows = []
    for group in usable_groups:
        responses = group.get("responses", [])
        rewards = group.get("rewards", [])
        advantages = group.get("advantages", [])
        best = _best_response(responses, rewards)
        rows.append(
            {
                "data_source": "gov_grpo_agent",
                "prompt": [{"role": "user", "content": group.get("prompt", "")}],
                "ability": "government_service_agent",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "case_id": group["case_id"],
                        "best_reward": best["reward"],
                        "expected_actions": best["actions"],
                    },
                },
                "extra_info": {
                    "case_id": group["case_id"],
                    "responses": responses,
                    "rewards": rewards,
                    "advantages": advantages,
                    "reward_mean": group.get("reward_mean"),
                    "reward_std": group.get("reward_std"),
                },
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(rows, output)

    report = _build_report(groups, rows, output)
    report_output = Path(report_path) if report_path else output.with_suffix(".report.json")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _best_response(responses, rewards):
    if not responses:
        return {"reward": 0.0, "actions": []}
    best_index = max(range(len(responses)), key=lambda index: rewards[index] if index < len(rewards) else 0.0)
    trajectory = responses[best_index].get("trajectory", [])
    return {
        "reward": float(rewards[best_index]) if best_index < len(rewards) else 0.0,
        "actions": [step.get("action") for step in trajectory if step.get("action")],
    }


def _read_jsonl(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _is_low_variance(group):
    return bool(group.get("low_variance")) or float(group.get("reward_std", 0.0)) == 0.0


def _write_parquet(rows, output):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to write verl parquet data. Install pandas and pyarrow.") from exc

    try:
        pd.DataFrame(rows).to_parquet(output, index=False)
    except ImportError as exc:
        raise RuntimeError("pyarrow or fastparquet is required to write verl parquet data.") from exc


def _build_report(groups, rows, output):
    total_responses = sum(len(group.get("responses", [])) for group in groups)
    usable_response_count = sum(len(row["extra_info"]["responses"]) for row in rows)
    low_variance_groups = sum(1 for group in groups if _is_low_variance(group))
    reward_values = [
        float(reward)
        for group in groups
        for reward in group.get("rewards", [])
    ]
    return {
        "groups": len(groups),
        "responses": total_responses,
        "usable_groups": len(rows),
        "usable_responses": usable_response_count,
        "low_variance_groups": low_variance_groups,
        "usable_group_rate": round(len(rows) / (len(groups) or 1), 6),
        "avg_reward": round(sum(reward_values) / len(reward_values), 6) if reward_values else 0.0,
        "best_reward": round(max(reward_values), 6) if reward_values else 0.0,
        "output": str(output),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert prepared GRPO JSONL into verl parquet data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)
    report = convert_grpo_jsonl_to_verl_parquet(args.input, args.output, args.report or None)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

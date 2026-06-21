import argparse
import json
from pathlib import Path


def prepare_grpo_dataset(input_path, output_path, report_path=None):
    groups = json.loads(Path(input_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output = Path(report_path) if report_path else output.with_suffix(".report.json")
    report_output.parent.mkdir(parents=True, exist_ok=True)

    total_responses = 0
    low_variance_groups = 0
    usable_groups = 0
    reward_sum = 0.0
    reward_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for group in groups:
            responses = group.get("responses", [])
            total_responses += len(responses)
            rewards = [float(response.get("reward", 0.0)) for response in responses]
            advantages = [float(response.get("advantage", 0.0)) for response in responses]
            reward_sum += sum(rewards)
            reward_count += len(rewards)
            low_variance = bool(group.get("low_variance")) or float(group.get("reward_std", 0.0)) == 0.0
            if low_variance:
                low_variance_groups += 1
            else:
                usable_groups += 1
            record = {
                "case_id": group["case_id"],
                "prompt": group.get("prompt", ""),
                "responses": [
                    {
                        "rollout_id": response.get("rollout_id"),
                        "trajectory": response.get("trajectory", []),
                    }
                    for response in responses
                ],
                "rewards": rewards,
                "advantages": advantages,
                "reward_mean": group.get("reward_mean"),
                "reward_std": group.get("reward_std"),
                "low_variance": low_variance,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "groups": len(groups),
        "responses": total_responses,
        "usable_groups": usable_groups,
        "low_variance_groups": low_variance_groups,
        "avg_reward": round(reward_sum / reward_count, 6) if reward_count else 0.0,
        "output": str(output),
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare grouped rollout data for GRPO training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)
    report = prepare_grpo_dataset(args.input, args.output, args.report or None)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

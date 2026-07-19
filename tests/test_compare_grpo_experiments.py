import json

from scripts.compare_grpo_experiments import paired_bootstrap_ci, read_case_metrics


def test_case_metrics_and_paired_bootstrap_detect_consistent_improvement(tmp_path):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_records = []
    candidate_records = []
    for index in range(20):
        case_id = f"case-{index}"
        for rollout in range(4):
            baseline_records.append(
                {
                    "case_id": case_id,
                    "environment_reward": 0.2,
                    "final_action_correct": float(rollout == 0),
                    "hard_gate": 0.0,
                    "required_tool_rate": 0.5,
                }
            )
            candidate_records.append(
                {
                    "case_id": case_id,
                    "environment_reward": 0.5,
                    "final_action_correct": float(rollout < 2),
                    "hard_gate": 0.0,
                    "required_tool_rate": 1.0,
                }
            )
    baseline_path.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_records), encoding="utf-8"
    )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_records), encoding="utf-8"
    )

    baseline = read_case_metrics(baseline_path)
    candidate = read_case_metrics(candidate_path)
    assert len(baseline) == len(candidate) == 20
    assert baseline["case-0"]["safe_success_at_k"] == 1.0
    assert candidate["case-0"]["process_success_at_k"] == 1.0
    ci_low, ci_high, paired_cases = paired_bootstrap_ci(
        baseline, candidate, "environment_reward", samples=1000
    )
    assert paired_cases == 20
    assert ci_low > 0
    assert ci_high > 0

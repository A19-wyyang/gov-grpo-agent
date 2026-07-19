from __future__ import annotations

import json

from scripts.export_grpo_metrics import load_rollouts


def test_rollout_export_includes_group_and_scenario_metrics(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    rows = []
    for case_id, rewards, scenario in (
        ("case-a", [0.5, 0.5, 0.5, 0.5], "success"),
        ("case-b", [0.0, 1.0, 0.0, 1.0], "risk"),
    ):
        for reward in rewards:
            rows.append(
                {
                    "step": 1,
                    "case_id": case_id,
                    "scenario_type": scenario,
                    "environment_reward": reward,
                    "score": reward,
                    "final_action_correct": float(reward > 0),
                    "hard_gate": float(reward == 0),
                    "unsafe_submit": 0.0,
                    "judge_used": 1.0,
                    "judge_score": 0.75,
                    "judge_clarity": 1.0,
                }
            )
    (rollout_dir / "1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    metrics, scenarios = load_rollouts(rollout_dir)

    assert metrics[1]["rollout_count"] == 8
    assert metrics[1]["group_count"] == 2
    assert metrics[1]["group_reward_std"] == 0.25
    assert metrics[1]["zero_variance_group_rate"] == 0.5
    assert metrics[1]["judge_clarity"] == 1.0
    assert scenarios["success"][1]["mean_reward"] == 0.5
    assert scenarios["risk"][1]["hard_gate_failure_rate"] == 0.5

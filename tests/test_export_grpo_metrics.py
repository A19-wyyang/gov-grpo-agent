from __future__ import annotations

import json

from scripts.export_grpo_metrics import (
    load_rollouts,
    length_reward_diagnostics,
    resolve_tensorboard_metrics,
    tool_call_stats,
    tool_name_stats,
)


def test_length_reward_diagnostics_detects_length_bias():
    diagnostics = length_reward_diagnostics(
        [
            {"output": "a", "environment_reward": 0.0},
            {"output": "bb", "environment_reward": 0.0},
            {"output": "ccc", "environment_reward": 1.0},
            {"output": "dddd", "environment_reward": 1.0},
        ]
    )
    assert diagnostics["mean_output_chars"] == 2.5
    assert diagnostics["reward_length_pearson"] > 0.8
    assert diagnostics["shortest_quartile_reward"] == 0.0
    assert diagnostics["longest_quartile_reward"] == 1.0
    assert diagnostics["long_minus_short_reward"] == 1.0


def test_tool_name_stats_detects_noncanonical_calls():
    text = """
    <tool_call>{"name":"government_service","arguments":{"action":"RISK_CHECK"}}</tool_call>
    <tool_call>{"name":"governmentService","arguments":{"action":"SUBMIT"}}</tool_call>
    """
    assert tool_name_stats(text) == (2, 1)


def test_tool_call_stats_detects_malformed_json():
    text = """
    <tool_call>{"name":"government_service","arguments":{"action":"RISK_CHECK"}}</tool_call>
    <tool_call>{"name":"government_service","arguments":{"action":"SUBMIT"}</tool_call>
    """
    assert tool_call_stats(text) == (2, 0, 1)


def test_tensorboard_metric_aliases_cover_verl_variants():
    scalars = {
        "actor/pg_loss": [(1, 0.1)],
        "actor/kl_loss": [(1, 0.02)],
        "actor/ppo_kl": [(1, 0.003)],
        "actor/entropy_loss": [(1, 0.5)],
        "actor/grad_norm": [(1, 0.7)],
        "actor/pg_clipfrac": [(1, 0.01)],
        "actor/pg_clipfrac_higher": [(1, 0.004)],
        "actor/pg_clipfrac_lower": [(1, 0.003)],
        "actor/lr": [(1, 5e-6)],
        "perf/max_memory_allocated_gb": [(1, 20.0)],
    }
    resolved, coverage = resolve_tensorboard_metrics(scalars)
    assert resolved["entropy"] == [(1, 0.5)]
    assert resolved["policy_update_kl"] == [(1, 0.003)]
    assert resolved["clip_fraction_higher"] == [(1, 0.004)]
    assert resolved["clip_fraction_lower"] == [(1, 0.003)]
    assert resolved["memory_allocated"] == [(1, 20.0)]
    assert coverage["missing_critical_metrics"] == []


def test_rollout_export_includes_group_and_scenario_metrics(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    rows = []
    for case_id, rewards, scenario in (
        ("case-a", [0.5, 0.5, 0.5, 0.5], "success"),
        ("case-b", [0.0, 1.0, 0.0, 1.0], "risk"),
    ):
        for rollout, reward in enumerate(rewards):
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
                    "output": (
                        f"output-{rollout % 2}"
                        if case_id == "case-b"
                        else "same-output"
                    ),
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
    assert metrics[1]["informative_group_count"] == 1.0
    assert metrics[1]["informative_group_rate"] == 0.5
    assert metrics[1]["informative_trajectory_count"] == 4.0
    assert metrics[1]["success_at_k"] == 1.0
    assert metrics[1]["safe_success_at_1"] == 0.75
    assert metrics[1]["process_success_at_1"] == 0.0
    assert metrics[1]["safe_success_at_k"] == 1.0
    assert metrics[1]["process_success_at_k"] == 0.0
    assert metrics[1]["unique_output_rate"] == 0.375
    assert metrics[1]["identical_output_group_rate"] == 0.5
    assert metrics[1]["mean_output_chars"] > 0
    assert metrics[1]["judge_clarity"] == 1.0
    assert scenarios["success"][1]["mean_reward"] == 0.5
    assert scenarios["risk"][1]["hard_gate_failure_rate"] == 0.5
    assert scenarios["success"][1]["identical_output_group_rate"] == 1.0

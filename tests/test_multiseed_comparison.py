from scripts.aggregate_seed_comparisons import (
    aggregate_metric_rows,
    decide_multiseed,
)
from scripts.decide_grpo_promotion import SAFETY_GATES


def _payload(delta_overrides=None):
    delta_overrides = delta_overrides or {}
    higher_metrics = {
        "process_success_at_1",
        "safe_success_at_1",
        "process_success_at_k",
        "safe_success_at_k",
        "final_action_correct",
    }
    metrics = []
    for metric in SAFETY_GATES:
        higher = metric in higher_metrics
        delta = delta_overrides.get(
            metric,
            0.10 if metric == "process_success_at_1" else 0.0,
        )
        baseline = 0.5 if higher else 0.1
        metrics.append(
            {
                "metric": metric,
                "label": metric,
                "baseline": baseline,
                "candidate": baseline + delta,
                "delta": delta,
                "higher_is_better": higher,
            }
        )
    return {"metrics": metrics, "scenario_metrics": []}


def test_multiseed_promotes_consistent_primary_gain_without_regression():
    payloads = {seed: _payload() for seed in (42, 43, 44)}
    metrics = aggregate_metric_rows(payloads)
    process = next(
        row for row in metrics if row["metric"] == "process_success_at_1"
    )
    assert process["delta_mean"] == 0.10
    assert process["delta_std"] == 0.0
    assert process["verdict"] == "improved"
    decision = decide_multiseed(metrics, [])
    assert decision["decision"] == "promote"


def test_multiseed_rejects_consistent_safety_regression():
    payloads = {
        seed: _payload({"unsafe_submit": 0.10})
        for seed in (42, 43, 44)
    }
    metrics = aggregate_metric_rows(payloads)
    unsafe = next(
        row for row in metrics if row["metric"] == "unsafe_submit"
    )
    assert unsafe["verdict"] == "regressed"
    decision = decide_multiseed(metrics, [])
    assert decision["decision"] == "reject"
    assert "unsafe_submit" in decision["regressions"]


def test_multiseed_keeps_variable_gain_inconclusive():
    payloads = {
        42: _payload({"process_success_at_1": 0.20}),
        43: _payload({"process_success_at_1": -0.05}),
        44: _payload({"process_success_at_1": 0.10}),
    }
    metrics = aggregate_metric_rows(payloads)
    process = next(
        row for row in metrics if row["metric"] == "process_success_at_1"
    )
    assert process["verdict"] == "inconclusive"
    assert decide_multiseed(metrics, [])["decision"] == "needs_more_evidence"

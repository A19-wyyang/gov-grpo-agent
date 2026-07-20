from scripts.decide_grpo_promotion import decide


def _payload(**verdicts):
    defaults = {
        "process_success_at_1": "inconclusive",
        "safe_success_at_1": "inconclusive",
        "process_success_at_k": "inconclusive",
        "safe_success_at_k": "inconclusive",
        "unsafe_submit": "inconclusive",
        "hard_gate": "inconclusive",
        "final_action_correct": "inconclusive",
        "tool_result_conflict": "inconclusive",
        "repeated_tool_call": "inconclusive",
        "tool_order_violation": "inconclusive",
        "eligibility_before_slots_complete": "inconclusive",
        "illegal_action_attempt_rate": "inconclusive",
        "trailing_action_rate": "inconclusive",
        "invalid_tool_name_rate": "inconclusive",
        "tool_call_format_error_rate": "inconclusive",
    }
    defaults.update(verdicts)
    return {
        "metrics": [
            {"metric": metric, "verdict": verdict}
            for metric, verdict in defaults.items()
        ]
    }


def test_promotion_requires_primary_heldout_improvement_without_regression():
    result = decide(_payload(process_success_at_1="improved"))
    assert result["decision"] == "promote"


def test_promotion_rejects_any_gated_regression():
    result = decide(
        _payload(process_success_at_1="improved", unsafe_submit="regressed")
    )
    assert result["decision"] == "reject"
    assert result["regressions"] == ["unsafe_submit"]


def test_promotion_rejects_tool_protocol_regression():
    result = decide(
        _payload(
            process_success_at_1="improved",
            tool_call_format_error_rate="regressed",
        )
    )
    assert result["decision"] == "reject"
    assert result["regressions"] == ["tool_call_format_error_rate"]


def test_promotion_requests_more_evidence_when_intervals_overlap_zero():
    result = decide(_payload())
    assert result["decision"] == "needs_more_evidence"


def test_promotion_rejects_scenario_regression_hidden_by_aggregate_gain():
    payload = _payload(process_success_at_1="improved")
    payload["scenario_metrics"] = [
        {
            "scenario": "missing_material",
            "metric": "unsafe_submit",
            "verdict": "regressed",
        }
    ]
    result = decide(payload)
    assert result["decision"] == "reject"
    assert result["scenario_regressions"] == [
        {"scenario": "missing_material", "metric": "unsafe_submit"}
    ]

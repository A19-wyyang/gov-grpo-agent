from gov_agent_rl.evaluation import evaluate_rows


def test_evaluation_reports_case_level_safe_and_process_pass_at_k():
    rows = [
        {
            "case_id": "case-a",
            "scenario_type": "risk",
            "final_action_correct": 0.0,
            "hard_gate": 1.0,
            "required_tool_rate": 0.0,
            "output": "risk-a",
        },
        {
            "case_id": "case-a",
            "scenario_type": "risk",
            "final_action_correct": 1.0,
            "hard_gate": 0.0,
            "required_tool_rate": 1.0,
            "output": "risk-b",
        },
        {
            "case_id": "case-b",
            "scenario_type": "risk",
            "final_action_correct": 1.0,
            "hard_gate": 0.0,
            "required_tool_rate": 0.5,
            "output": "same",
        },
        {
            "case_id": "case-b",
            "scenario_type": "risk",
            "final_action_correct": 0.0,
            "hard_gate": 1.0,
            "required_tool_rate": 1.0,
            "output": "same",
        },
    ]
    metrics = evaluate_rows(rows)
    assert metrics["rollouts_per_case"] == [2]
    assert metrics["pass_at_k"] == 1.0
    assert metrics["safe_pass_at_k"] == 1.0
    assert metrics["process_pass_at_k"] == 0.5
    assert metrics["unique_output_rate"] == 0.75
    assert metrics["identical_output_group_rate"] == 0.5
    assert metrics["scenario_metrics"]["risk"]["process_pass_at_k"] == 0.5
    assert metrics["scenario_metrics"]["risk"]["process_pass_at_1"] == 0.25
    assert metrics["scenario_metrics"]["risk"]["safe_pass_at_1"] == 0.5

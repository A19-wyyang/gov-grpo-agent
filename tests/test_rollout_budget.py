from PIL import Image

from scripts.analyze_rollout_budget import analyze_records, draw_analysis


def _record(case_id, success):
    return {
        "case_id": case_id,
        "final_action_correct": float(success),
        "hard_gate": 0.0 if success else 1.0,
        "required_tool_rate": 1.0 if success else 0.0,
    }


def test_rollout_budget_recommends_more_samples_for_sparse_mixed_success():
    records = []
    for index in range(40):
        # One success in four produces useful but sparse group contrast; a
        # larger group materially raises both coverage and mixed outcomes.
        records.extend(
            _record(f"case-{index}", rollout == 0)
            for rollout in range(4)
        )
    result = analyze_records(records, [4, 8, 16], bootstrap_samples=100)
    assert result["current_n"] == 4
    assert result["recommendation"]["action"] == "run_rollout_ablation"
    assert result["recommendation"]["recommended_n"] == 8


def test_rollout_budget_keeps_n_when_every_case_has_zero_success():
    records = [
        _record(f"case-{index}", False)
        for index in range(20)
        for _ in range(4)
    ]
    result = analyze_records(records, [4, 8, 16], bootstrap_samples=100)
    assert result["recommendation"]["action"] == "keep_current_n"
    process = [
        row for row in result["metrics"]
        if row["metric"] == "process_success" and row["target_n"] == 16
    ][0]
    assert process["projected_success_at_n"] == 0.0


def test_rollout_budget_visualization_is_renderable(tmp_path):
    records = [
        _record(f"case-{index}", rollout == 0)
        for index in range(10)
        for rollout in range(4)
    ]
    payload = analyze_records(records, [4, 8, 16], bootstrap_samples=50)
    output = tmp_path / "rollout_budget_analysis.png"
    draw_analysis(payload, output)
    with Image.open(output) as image:
        assert image.size == (1400, 850)

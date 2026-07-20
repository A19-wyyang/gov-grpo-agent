from gov_agent_rl.sft_reporting import (
    extract_sft_series,
    render_scenario_losses,
    render_sft_training,
    scenario_losses,
)


def test_sft_reporting_extracts_loss_lr_and_scenario_metrics(tmp_path):
    history = [
        {"step": 1, "loss": 1.2, "learning_rate": 2e-5},
        {"step": 25, "loss": 0.6, "learning_rate": 1e-5},
        {"step": 25, "eval_loss": 0.7},
    ]
    series = extract_sft_series(history)
    assert series["train_loss"] == [(1, 1.2), (25, 0.6)]
    assert series["eval_loss"] == [(25, 0.7)]
    assert series["learning_rate"] == [(1, 2e-5), (25, 1e-5)]

    scenario_eval = {
        "risk": {"eval_risk_loss": 0.4},
        "missing_information": {
            "eval_missing_information_loss": 0.9
        },
    }
    assert scenario_losses(scenario_eval) == {
        "risk": 0.4,
        "missing_information": 0.9,
    }
    training_png = tmp_path / "training.png"
    scenario_png = tmp_path / "scenario.png"
    render_sft_training(history, training_png)
    render_scenario_losses(scenario_eval, scenario_png)
    assert training_png.stat().st_size > 0
    assert scenario_png.stat().st_size > 0

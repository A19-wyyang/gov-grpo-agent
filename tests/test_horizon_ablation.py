from pathlib import Path

import yaml


def test_horizon_ablation_changes_only_max_steps():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load(
        (root / "configs/tools/government_service.yaml").read_text(encoding="utf-8")
    )
    horizon10 = yaml.safe_load(
        (root / "configs/tools/government_service_h10.yaml").read_text(encoding="utf-8")
    )
    baseline_tool = baseline["tools"][0]
    horizon_tool = horizon10["tools"][0]
    assert baseline_tool["config"]["max_steps"] == 8
    assert horizon_tool["config"]["max_steps"] == 10

    baseline_tool["config"]["max_steps"] = 10
    assert baseline == horizon10


def test_horizon10_reference_flows_keep_two_recovery_actions():
    from gov_agent_rl.data_builder import build_cases

    longest = max(len(case.reference_actions) for case in build_cases())
    assert longest == 8
    assert 10 - longest == 2

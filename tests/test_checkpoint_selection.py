import json
import sys

import pytest

from gov_agent_rl.data_builder import build_cases
from scripts.select_best_grpo_checkpoint import (
    group_complete_records,
    main,
    selection_key,
    summarize_rescored,
    wilson_lower,
)


def _safe_output(case):
    calls = []
    for slot in case.rules.required_slots:
        if slot not in case.visible_slots:
            calls.append({"action": "ASK_USER", "slot": slot})
    calls.extend(
        [
            {"action": "POLICY_SEARCH", "query": case.title},
            {"action": "ELIGIBILITY_CHECK"},
            {"action": "MATERIAL_CHECK"},
            {"action": "RISK_CHECK"},
            {
                "action": case.expected_result.final_action.value,
                "message": case.expected_result.reason,
            },
        ]
    )
    return "\n".join(
        f'<tool_call>{json.dumps({"name": "government_service", "arguments": call})}</tool_call>'
        for call in calls
    )


def test_wilson_lower_is_monotonic():
    assert wilson_lower(80, 100) > wilson_lower(70, 100)


def test_checkpoint_summary_reports_pass_at_1_and_pass_at_k_separately():
    records = []
    for case_id in ("case-a", "case-b"):
        for rollout in range(4):
            records.append(
                {
                    "case_id": case_id,
                    "scenario_type": (
                        "missing_information" if case_id == "case-a" else "risk"
                    ),
                    "final_action_correct": float(rollout == 0),
                    "hard_gate": 0.0,
                    "required_tool_rate": 1.0,
                    "unsafe_submit": 0.0,
                    "illegal_action_attempt_rate": 0.0,
                    "environment_reward": float(rollout == 0),
                }
            )
    summary = summarize_rescored(records, {"case-a", "case-b"})
    assert summary["cases"] == 2
    assert summary["rollouts_per_case"] == 4
    assert summary["process_pass_at_1"] == 0.25
    assert summary["process_pass_at_1_lcb95"] == wilson_lower(2, 8)
    assert summary["safe_pass_at_1"] == 0.25
    assert summary["process_pass_at_k"] == 1.0
    assert summary["process_pass_at_k_lcb95"] == wilson_lower(2, 2)
    assert summary["process_success"] == 1.0
    assert summary["process_success_lcb95"] == wilson_lower(2, 2)
    assert summary["worst_scenario_process_pass_at_1"] == 0.25
    assert summary["worst_scenario_process_pass_at_1_lcb95"] == wilson_lower(1, 4)


def test_checkpoint_selection_prioritizes_weakest_scenario_over_mean():
    high_mean_brittle = {
        "worst_scenario_process_pass_at_1_lcb95": 0.10,
        "process_pass_at_1_lcb95": 0.80,
        "safe_pass_at_1_lcb95": 0.80,
        "unsafe_submit": 0.0,
        "illegal_action_attempt_rate": 0.0,
        "process_pass_at_k_lcb95": 0.90,
        "safe_pass_at_k_lcb95": 0.90,
        "final_action_correct": 0.95,
        "environment_reward": 0.90,
    }
    lower_mean_robust = {
        **high_mean_brittle,
        "worst_scenario_process_pass_at_1_lcb95": 0.30,
        "process_pass_at_1_lcb95": 0.70,
        "final_action_correct": 0.85,
        "environment_reward": 0.80,
    }
    assert max(
        (high_mean_brittle, lower_mean_robust),
        key=selection_key,
    ) is lower_mean_robust


def test_checkpoint_selection_prefers_pass_at_1_over_pass_at_k():
    strong_at_k_only = {
        "worst_scenario_process_pass_at_1_lcb95": 0.10,
        "process_pass_at_1_lcb95": 0.20,
        "safe_pass_at_1_lcb95": 0.20,
        "unsafe_submit": 0.0,
        "illegal_action_attempt_rate": 0.0,
        "process_pass_at_k_lcb95": 0.90,
        "safe_pass_at_k_lcb95": 0.90,
        "final_action_correct": 0.25,
        "environment_reward": 0.25,
    }
    deployable_at_1 = {
        **strong_at_k_only,
        "worst_scenario_process_pass_at_1_lcb95": 0.30,
        "process_pass_at_1_lcb95": 0.40,
        "safe_pass_at_1_lcb95": 0.40,
        "process_pass_at_k_lcb95": 0.70,
        "safe_pass_at_k_lcb95": 0.70,
    }
    assert max(
        (strong_at_k_only, deployable_at_1), key=selection_key
    ) is deployable_at_1


def test_checkpoint_summary_rejects_partial_or_nonuniform_case_coverage():
    with pytest.raises(ValueError, match="incomplete case coverage"):
        group_complete_records([{"case_id": "case-a"}], {"case-a", "case-b"})
    with pytest.raises(ValueError, match="non-uniform rollouts"):
        group_complete_records(
            [
                {"case_id": "case-a"},
                {"case_id": "case-a"},
                {"case_id": "case-b"},
            ],
            {"case-a", "case-b"},
        )


def test_checkpoint_selector_prefers_process_safe_step(tmp_path, monkeypatch):
    cases = [
        case
        for case in build_cases()
        if case.expected_result.final_action.value == "REFUSE"
    ][:2]
    validation_dir = tmp_path / "validation"
    checkpoint_root = tmp_path / "checkpoints"
    output_dir = tmp_path / "selection"
    validation_dir.mkdir()
    for step in (5, 10):
        (checkpoint_root / f"global_step_{step}").mkdir(parents=True)
    safe_rows = [
        {"case_id": case.case_id, "output": _safe_output(case), "judge_used": 0.0}
        for case in cases
    ]
    unsafe_rows = [
        {
            "case_id": case.case_id,
            "output": (
                '<tool_call>{"name":"government_service","arguments":'
                '{"action":"SUBMIT","message":"直接提交"}}</tool_call>'
            ),
            "judge_used": 0.0,
        }
        for case in cases
    ]
    for step, rows in ((5, safe_rows), (10, unsafe_rows)):
        (validation_dir / f"{step}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_best_grpo_checkpoint.py",
            "--validation-dir",
            str(validation_dir),
            "--cases",
            str(cases_path),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output-dir",
            str(output_dir),
        ],
    )
    main()
    result = json.loads((output_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert result["best_step"] == 5
    assert (output_dir / "checkpoint_selection.png").is_file()

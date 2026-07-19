import json
import sys

from gov_agent_rl.data_builder import build_cases
from scripts.select_best_grpo_checkpoint import main, wilson_lower


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

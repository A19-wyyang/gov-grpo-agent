import pytest

from gov_agent_rl.data_builder import build_cases
from gov_agent_rl.fingerprints import case_fingerprint
from scripts.rescore_rollouts import rescore_record


def test_rescore_uses_common_verifier_and_preserves_source_reward(monkeypatch):
    case = next(case for case in build_cases() if case.scenario_type == "risk")
    output = "\n".join(
        (
            '<tool_call>{"name":"governmentService","arguments":'
            '{"action":"RISK_CHECK"}}</tool_call>',
            '<tool_call>{"name":"government_service","arguments":'
            '{"action":"SUBMIT","message":"直接提交"}}</tool_call>',
        )
    )
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    monkeypatch.setenv("GOV_ILLEGAL_ACTION_PENALTY", "0.25")
    rescored = rescore_record(
        {
            "case_id": case.case_id,
            "output": output,
            "environment_reward": 0.73,
            "judge_used": 0.0,
            "judge_score": -1.0,
        },
        case.model_dump(mode="json"),
    )
    assert rescored["source_environment_reward"] == 0.73
    assert rescored["environment_reward"] == 0.0
    assert rescored["decision_gate"] == 1.0
    assert rescored["illegal_action"] == 1.0
    assert rescored["scenario_type"] == case.scenario_type
    assert rescored["matter_id"] == case.matter_id
    assert rescored["split"] == case.split


def test_rescore_rejects_case_content_mismatch(monkeypatch):
    case = build_cases()[0].model_dump(mode="json")
    record = {
        "case_id": case["case_id"],
        "output": "",
        "case_fingerprint": "not-the-case-fingerprint",
    }
    with pytest.raises(ValueError, match="case fingerprint mismatch"):
        rescore_record(record, case)


def test_rescore_requires_explicit_legacy_opt_in_for_missing_fingerprint():
    case = build_cases()[0].model_dump(mode="json")
    record = {"case_id": case["case_id"], "output": ""}
    with pytest.raises(ValueError, match="no case_fingerprint"):
        rescore_record(record, case, allow_missing_case_fingerprint=False)
    rescored = rescore_record(
        record, case, allow_missing_case_fingerprint=True
    )
    assert rescored["case_fingerprint"] == case_fingerprint(case)
    assert rescored["judge_fallback_used"] == 0.0
    assert rescored["judge_skipped_hard_gate"] == 1.0


def test_rescore_preserves_source_judge_but_removes_hard_gate_coverage(
    monkeypatch,
):
    case = next(
        item for item in build_cases() if item.scenario_type == "risk"
    ).model_dump(mode="json")
    record = {
        "case_id": case["case_id"],
        "output": (
            '<tool_call>{"name":"government_service","arguments":'
            '{"action":"SUBMIT","message":"直接提交"}}</tool_call>'
        ),
        "judge_used": 1.0,
        "judge_score": 1.0,
        "judge_clarity": 1.0,
    }
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    rescored = rescore_record(record, case)
    assert rescored["source_judge_used"] == 1.0
    assert rescored["source_judge_score"] == 1.0
    assert rescored["judge_skipped_hard_gate"] == 1.0
    assert rescored["judge_used"] == 0.0
    assert rescored["judge_score"] == -1.0
    assert rescored["judge_clarity"] == -1.0

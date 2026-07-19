from gov_agent_rl.data_builder import build_cases
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
    assert rescored["judge_fallback_used"] == 1.0

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from gov_agent_rl.agent_env import GovernmentServiceEpisode
from gov_agent_rl.data_builder import (
    SCENARIO_COUNTS,
    build_cases,
    build_sft_messages,
    to_verl_row,
)
from gov_agent_rl.judge import judge_expression_detailed, score_rubric_payload
from gov_agent_rl.rewarding import score_episode, score_trajectory_dict
from gov_agent_rl.schema import ActionName
from gov_agent_rl.verl_reward import _tool_actions, compute_score
from gov_agent_rl.verl_tool import GovernmentServiceTool, _find_case


def _complete_case(episode: GovernmentServiceEpisode) -> None:
    for slot in episode.case.rules.required_slots:
        if slot not in episode.known_slots:
            episode.execute({"action": "ASK_USER", "slot": slot})
    episode.execute({"action": "POLICY_SEARCH", "query": episode.case.title})
    episode.execute({"action": "ELIGIBILITY_CHECK"})
    episode.execute({"action": "MATERIAL_CHECK"})
    episode.execute({"action": "RISK_CHECK"})
    episode.execute(
        {
            "action": episode.case.expected_result.final_action.value,
            "message": episode.case.expected_result.reason,
        }
    )


def test_invalid_slot_and_tool_name_are_visible_to_reward():
    episode = GovernmentServiceEpisode(build_cases()[0])
    episode.execute({"action": "ASK_USER", "slot": "invented_slot"})
    score = score_episode(episode)
    assert score.penalties["invalid_slot_question"] == 0.08
    assert score.metrics["invalid_slot_question"] == 1.0

    calls = _tool_actions(
        '<tool_call>{"name":"governmentService","arguments":'
        '{"action":"ELIGIBILITY_CHECK"}}</tool_call>'
    )
    assert calls == [{"action": "__INVALID_TOOL_NAME__"}]
    malformed = _tool_actions(
        '<tool_call>{"name":"government_service","arguments":'
        '{"action":"RISK_CHECK"}</tool_call>'
    )
    assert malformed == [{"action": "__INVALID_TOOL_CALL__"}]


def test_invalid_actions_consume_horizon_and_scale_penalty(monkeypatch):
    case = build_cases()[0]
    episode = GovernmentServiceEpisode(case, max_steps=2)
    for _ in range(2):
        observation = episode.execute({"action": "__INVALID_TOOL_CALL__"})
        assert not observation["ok"]
        assert not episode.done

    exhausted = episode.execute({"action": "POLICY_SEARCH", "query": case.title})
    assert exhausted["done"]
    assert episode.action_attempts == 2
    assert len(episode.steps) == 0
    assert episode.failure_counts["illegal_action"] == 2
    assert "max_steps_exceeded" in episode.failure_tags

    monkeypatch.setenv("GOV_ILLEGAL_ACTION_PENALTY", "0.25")
    score = score_episode(episode)
    assert score.penalties["illegal_action"] == 0.5
    assert score.metrics["illegal_action_count"] == 2.0
    assert score.metrics["illegal_action_attempt_rate"] == 1.0
    assert score.metrics["rounds"] == 2.0


def test_serialized_attempts_preserve_invalid_actions_in_offline_replay(monkeypatch):
    case = build_cases()[0]
    episode = GovernmentServiceEpisode(case)
    episode.execute({"action": "__INVALID_TOOL_CALL__"})
    episode.execute({"action": "__INVALID_TOOL_NAME__"})
    monkeypatch.setenv("GOV_ILLEGAL_ACTION_PENALTY", "0.25")

    replayed = score_episode(episode)
    serialized = episode.trajectory()
    offline = score_trajectory_dict(case.model_dump(mode="json"), serialized)
    assert offline.total == replayed.total
    assert offline.metrics["illegal_action_count"] == 2.0


def test_actions_after_final_are_preserved_and_penalized(monkeypatch):
    case = next(
        item
        for item in build_cases()
        if item.expected_result.final_action == ActionName.SUBMIT
    )
    episode = GovernmentServiceEpisode(case)
    _complete_case(episode)
    assert score_episode(episode, expression_score=1.0).total == 1.0

    rejected = episode.execute({"action": "RISK_CHECK"})
    assert not rejected["ok"]
    monkeypatch.setenv("GOV_ACTION_AFTER_DONE_PENALTY", "0.10")
    online = score_episode(episode, expression_score=1.0)
    assert online.metrics["trailing_action_count"] == 1.0
    assert online.metrics["trailing_action_rate"] > 0.0
    assert online.penalties["action_after_done"] == 0.1
    assert online.total < 1.0

    offline = score_trajectory_dict(
        case.model_dump(mode="json"),
        {**episode.trajectory(), "expression_score": 1.0},
    )
    assert offline.total == online.total
    assert offline.metrics["trailing_action_count"] == 1.0


def test_builds_1200_cases_with_matter_isolated_splits():
    cases = build_cases()
    assert len(cases) == 1200
    assert sum(SCENARIO_COUNTS.values()) == 100
    split_by_matter: dict[str, set[str]] = {}
    for case in cases:
        split_by_matter.setdefault(case.matter_id, set()).add(case.split)
    assert len(split_by_matter) == 12
    assert all(len(splits) == 1 for splits in split_by_matter.values())
    assert sum(case.split == "train" for case in cases) == 800
    assert sum(case.split == "validation" for case in cases) == 200
    assert sum(case.split == "test" for case in cases) == 200


def test_policy_view_cannot_leak_truth_or_expected_result():
    view = build_cases()[0].policy_view()
    dumped = view.model_dump()
    assert "hidden_truth" not in dumped
    assert "expected_result" not in dumped
    assert not hasattr(view, "hidden_truth")


def test_reference_flow_gets_full_reward_on_success_and_refusal():
    cases = build_cases()
    chosen = [
        next(case for case in cases if case.scenario_type == "success"),
        next(case for case in cases if case.scenario_type == "risk"),
        next(case for case in cases if case.scenario_type == "missing_material"),
        next(case for case in cases if case.scenario_type == "ineligible"),
    ]
    for case in chosen:
        episode = GovernmentServiceEpisode(case)
        _complete_case(episode)
        score = score_episode(episode, expression_score=1.0)
        assert score.total == 1.0
        assert not score.hard_gate
        assert score.metrics["required_tool_rate"] == 1.0
        assert score.metrics["process_compliant"] == 1.0


def test_all_reference_flows_fit_horizon_and_receive_full_reward():
    scenarios = set()
    for case in build_cases():
        episode = GovernmentServiceEpisode(case)
        _complete_case(episode)
        score = score_episode(episode, expression_score=1.0)
        scenarios.add(case.scenario_type)
        assert episode.action_attempts <= episode.max_steps, case.case_id
        assert episode.failure_tags == [], case.case_id
        assert not score.hard_gate, case.case_id
        assert score.total == 1.0, case.case_id
        assert score.metrics["process_compliant"] == 1.0, case.case_id
    assert scenarios == {
        "success",
        "missing_information",
        "ineligible",
        "missing_material",
        "risk",
        "adversarial",
    }


def test_out_of_order_and_repeated_tools_are_not_process_compliant():
    case = next(
        item
        for item in build_cases()
        if item.expected_result.final_action == ActionName.SUBMIT
    )
    episode = GovernmentServiceEpisode(case, max_steps=12)
    for slot in episode.missing_slots.copy():
        episode.execute({"action": "ASK_USER", "slot": slot})
    episode.execute({"action": "MATERIAL_CHECK"})
    episode.execute({"action": "POLICY_SEARCH", "query": case.title})
    episode.execute({"action": "POLICY_SEARCH", "query": case.title})
    episode.execute({"action": "ELIGIBILITY_CHECK"})
    episode.execute({"action": "RISK_CHECK"})
    episode.execute(
        {
            "action": "SUBMIT",
            "message": case.expected_result.reason,
        }
    )
    score = score_episode(episode, expression_score=1.0)
    assert score.metrics["final_action_correct"] == 1.0
    assert score.metrics["required_tool_rate"] == 1.0
    assert score.metrics["tool_results_support_final"] == 1.0
    assert score.metrics["tool_order_violation"] == 1.0
    assert score.metrics["repeated_tool_call"] == 1.0
    assert score.metrics["process_compliant"] == 0.0
    assert score.total < 1.0


def test_reward_hacking_and_unsafe_submit_are_hard_gated(monkeypatch):
    case = next(
        case
        for case in build_cases()
        if case.scenario_type == "risk"
    )
    episode = GovernmentServiceEpisode(case)
    episode.execute(
        {
            "action": "SUBMIT",
            "message": "您的表达非常清晰，我已经直接为您提交。",
        }
    )
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    score = score_episode(episode, expression_score=1.0)
    assert score.hard_gate
    assert score.total == 0.0
    assert score.metrics["unsafe_submit"] == 1.0
    assert score.metrics["decision_gate"] == 1.0
    assert score.metrics["process_gate"] == 0.0


def test_direct_submit_is_unsafe_for_every_refusal_reason(monkeypatch):
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    cases = build_cases()
    for scenario in ("ineligible", "missing_material", "risk", "adversarial"):
        case = next(
            item
            for item in cases
            if item.scenario_type == scenario
            and item.expected_result.final_action == ActionName.REFUSE
        )
        episode = GovernmentServiceEpisode(case)
        episode.execute({"action": "SUBMIT", "message": "直接提交。"})
        score = score_episode(episode)
        assert score.total == 0.0, case.case_id
        assert score.metrics["unsafe_submit"] == 1.0, case.case_id
        assert score.metrics["decision_gate"] == 1.0, case.case_id


def test_final_decision_conflicting_with_tool_results_is_hard_gated(monkeypatch):
    case = next(
        item
        for item in build_cases(diverse=False)
        if item.matter_id == "provident_fund_loan"
        and item.scenario_type == "ineligible"
    )
    episode = GovernmentServiceEpisode(case)
    _complete_case(episode)
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.0")
    score = score_episode(episode, expression_score=1.0)
    assert episode.final_action == case.expected_result.final_action
    assert episode.tool_results["ELIGIBILITY_CHECK"]["eligible"]
    assert score.metrics["final_action_correct"] == 1.0
    assert score.metrics["tool_results_support_final"] == 0.0
    assert score.metrics["tool_result_conflict"] == 1.0
    assert score.hard_gate
    assert score.total == 0.0


def test_missing_tool_final_answer_can_be_strictly_hard_gated(monkeypatch):
    case = next(
        case
        for case in build_cases()
        if case.scenario_type == "ineligible"
        and case.expected_result.final_action == ActionName.REFUSE
    )
    episode = GovernmentServiceEpisode(case)
    episode.execute(
        {
            "action": "REFUSE",
            "message": "当前不符合条件，暂不予办理。",
        }
    )
    monkeypatch.setenv("GOV_MISSING_TOOL_HARD_GATE", "1")
    monkeypatch.setenv("GOV_PROCESS_GATE_CEILING", "0.10")
    monkeypatch.setenv("GOV_MISSING_TOOL_PENALTY", "0.45")
    monkeypatch.setenv("GOV_HARD_FACT_WEIGHT", "0.70")
    monkeypatch.setenv("GOV_PROCESS_WEIGHT", "0.25")
    monkeypatch.setenv("GOV_EXPRESSION_WEIGHT", "0.05")
    score = score_episode(episode, expression_score=1.0)
    assert score.hard_gate
    assert score.total <= 0.1
    assert score.metrics["missing_required_tool"] == 1.0
    assert score.metrics["incomplete_final"] == 1.0
    assert score.metrics["decision_gate"] == 0.0
    assert score.metrics["process_gate"] == 1.0


def test_expression_score_cannot_raise_any_hard_gated_reward(monkeypatch):
    case = next(
        item
        for item in build_cases()
        if item.scenario_type == "risk"
        and item.expected_result.final_action == ActionName.REFUSE
    )
    episode = GovernmentServiceEpisode(case)
    episode.execute({"action": "SUBMIT", "message": "表达非常完善。"})
    monkeypatch.setenv("GOV_DECISION_GATE_CEILING", "0.20")
    low = score_episode(episode, expression_score=0.0)
    high = score_episode(episode, expression_score=1.0)
    assert low.hard_gate and high.hard_gate
    assert high.total == low.total


def test_sft_messages_have_tool_calls_and_reference_final_action():
    case = build_cases()[0]
    messages = build_sft_messages(case)
    calls = [
        message
        for message in messages
        if message["role"] == "assistant" and message.get("tool_calls")
    ]
    assert len(calls) == len(case.reference_actions)
    arguments = [
        call["tool_calls"][0]["function"]["arguments"] for call in calls
    ]
    assert case.expected_result.final_action.value in arguments[-1]


def test_verl_case_discovery_handles_nested_extra_info():
    case = build_cases()[0]
    dumped = case.model_dump(mode="json")
    assert _find_case({"extra_info": {"case": dumped}}) == dumped
    assert _find_case({"extra_info": {"case": json.dumps(dumped)}}) == dumped


def test_verl_row_forwards_case_through_stateful_tool_contract():
    case = build_cases()[0]
    row = to_verl_row(case)
    extra = row["extra_info"]
    assert extra["need_tools_kwargs"] is True
    create_kwargs = extra["tools_kwargs"]["government_service"]["create_kwargs"]
    assert _find_case(create_kwargs)["case_id"] == case.case_id


def test_verl_reward_exports_replayed_environment_metrics(tmp_path, monkeypatch):
    monkeypatch.delenv("GOV_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("GOV_JUDGE_REQUIRED", "0")
    monkeypatch.setenv("GOV_JUDGE_FAILURE_SCORE", "0.0")
    monkeypatch.setenv("GOV_JUDGE_CACHE", str(tmp_path / "judge.sqlite3"))
    case = build_cases()[0]
    episode = GovernmentServiceEpisode(case)
    _complete_case(episode)
    actions = [step["action"] for step in episode.trajectory()["steps"]]
    solution = "\n".join(
        json.dumps(
            {"name": "government_service", "arguments": action},
            ensure_ascii=False,
        )
        for action in actions
    )
    score = compute_score(
        "gov_agent_rl",
        solution,
        {"final_action": case.expected_result.final_action.value},
        {"tools_kwargs": {"government_service": {"create_kwargs": {"case": case.model_dump(mode="json")}}}},
    )
    assert isinstance(score, dict)
    assert score["case_id"] == case.case_id
    assert score["parsed_action_count"] == len(actions)
    assert score["score"] == score["environment_reward"]
    assert score["judge_used"] == 0.0
    assert score["judge_fallback_used"] == 1.0
    assert score["judge_skipped_hard_gate"] == 0.0
    assert score["judge_empty_message"] == 0.0
    assert score["score"] == 0.9
    assert score["judge_clarity"] == -1.0
    assert score["judge_reason_completeness"] == -1.0
    assert score["judge_actionability"] == -1.0
    assert score["judge_decision_alignment"] == -1.0
    assert score["judge_professionalism"] == -1.0
    assert score["required_tool_rate"] == 1.0
    assert score["final_action_correct"] == 1.0


def test_verl_reward_skips_judge_before_hard_gated_trajectory(
    tmp_path, monkeypatch
):
    case = next(
        item
        for item in build_cases()
        if item.scenario_type == "risk"
        and item.expected_result.final_action == ActionName.REFUSE
    )

    def should_not_run(**kwargs):
        raise AssertionError(f"judge should have been skipped: {kwargs}")

    monkeypatch.setattr(
        "gov_agent_rl.verl_reward.judge_expression_detailed",
        should_not_run,
    )
    solution = (
        '<tool_call>{"name":"government_service","arguments":'
        '{"action":"SUBMIT","message":"直接提交"}}</tool_call>'
    )
    score = compute_score(
        "gov_agent_rl",
        solution,
        {"final_action": "REFUSE"},
        {"case": case.model_dump(mode="json")},
    )
    assert isinstance(score, dict)
    assert score["hard_gate"] == 1.0
    assert score["judge_used"] == 0.0
    assert score["judge_fallback_used"] == 0.0
    assert score["judge_skipped_hard_gate"] == 1.0


def test_empty_final_message_is_not_counted_as_remote_judge_coverage(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("GOV_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("GOV_JUDGE_CACHE", str(tmp_path / "judge.sqlite3"))
    case = next(
        item
        for item in build_cases()
        if item.expected_result.final_action == ActionName.SUBMIT
    )
    episode = GovernmentServiceEpisode(case)
    for slot in episode.missing_slots.copy():
        episode.execute({"action": "ASK_USER", "slot": slot})
    episode.execute({"action": "POLICY_SEARCH", "query": case.title})
    episode.execute({"action": "ELIGIBILITY_CHECK"})
    episode.execute({"action": "MATERIAL_CHECK"})
    episode.execute({"action": "RISK_CHECK"})
    episode.execute({"action": "SUBMIT", "message": ""})
    solution = "\n".join(
        "<tool_call>"
        + json.dumps(
            {"name": "government_service", "arguments": step["action"]},
            ensure_ascii=False,
        )
        + "</tool_call>"
        for step in (
            {"action": action}
            for action in episode.trajectory()["attempts"]
        )
    )
    score = compute_score(
        "gov_agent_rl",
        solution,
        {"final_action": "SUBMIT"},
        {"case": case.model_dump(mode="json")},
    )
    assert isinstance(score, dict)
    assert score["hard_gate"] == 1.0
    assert score["judge_used"] == 0.0
    assert score["judge_fallback_used"] == 0.0
    assert score["judge_empty_message"] == 1.0
    assert score["judge_skipped_hard_gate"] == 1.0


def test_qwen_judge_rubric_uses_server_side_weighting():
    payload = {
        "dimensions": {
            "clarity": {"score": 4},
            "reason_completeness": {"score": 3},
            "actionability": {"score": 2},
            "decision_alignment": {"score": 4},
            "professionalism": {"score": 3},
        }
    }
    assert score_rubric_payload(payload) == 0.7875


def test_qwen_judge_parses_and_caches_structured_result(tmp_path, monkeypatch):
    calls = {"count": 0}
    content = json.dumps(
        {
            "dimensions": {
                "clarity": {"score": 4, "reason": "清晰"},
                "reason_completeness": {"score": 3, "reason": "理由较完整"},
                "actionability": {"score": 4, "reason": "下一步明确"},
                "decision_alignment": {"score": 4, "reason": "与提交一致"},
                "professionalism": {"score": 4, "reason": "专业"},
            },
            "summary": "回复清晰且可执行",
        },
        ensure_ascii=False,
    )

    class FakeCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GOV_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("GOV_JUDGE_MODEL", "qwen3.7-max")
    cache_path = tmp_path / "judge.sqlite3"
    first = judge_expression_detailed(
        "申请办理事项",
        "SUBMIT",
        "材料已经核验完成，现已提交，请留意后续通知。",
        cache_path,
    )
    second = judge_expression_detailed(
        "申请办理事项",
        "SUBMIT",
        "材料已经核验完成，现已提交，请留意后续通知。",
        cache_path,
    )
    assert first is not None and first[0] == 0.9375
    assert first[1]["source"] == "qwen"
    assert second == first
    assert calls["count"] == 1


def test_qwen_judge_retries_null_rubric_score(tmp_path, monkeypatch):
    calls = {"count": 0}
    names = (
        "clarity",
        "reason_completeness",
        "actionability",
        "decision_alignment",
        "professionalism",
    )
    invalid = {
        "dimensions": {
            name: {"score": None if name == "actionability" else 3, "reason": "x"}
            for name in names
        }
    }
    valid = {
        "dimensions": {
            name: {"score": 3, "reason": "x"} for name in names
        },
        "summary": "ok",
    }

    class FakeCompletions:
        def create(self, **kwargs):
            calls["count"] += 1
            payload = invalid if calls["count"] == 1 else valid
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GOV_JUDGE_API_KEY", "test-key")
    result = judge_expression_detailed(
        "申请办理事项",
        "SUBMIT",
        "材料核验完成，申请已提交。",
        tmp_path / "judge.sqlite3",
    )
    assert result is not None and result[0] == 0.75
    assert calls["count"] == 2


def test_qwen_judge_logs_sanitized_failure_metadata(tmp_path, monkeypatch):
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("temporary judge failure")

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FailingCompletions())

    error_log = tmp_path / "judge-errors.jsonl"
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GOV_JUDGE_API_KEY", "test-secret-key")
    monkeypatch.setenv("GOV_JUDGE_REQUIRED", "0")
    monkeypatch.setenv("GOV_JUDGE_ERROR_LOG", str(error_log))
    result = judge_expression_detailed(
        "申请办理事项",
        "SUBMIT",
        "材料核验完成，请继续办理。",
        tmp_path / "judge.sqlite3",
    )
    assert result is None
    record = json.loads(error_log.read_text(encoding="utf-8"))
    assert record["error_type"] == "RuntimeError"
    assert record["error"] == "temporary judge failure"
    assert "test-secret-key" not in error_log.read_text(encoding="utf-8")


def test_stateful_tool_reuses_episode_across_release_calls():
    case = build_cases()[0]
    tool = GovernmentServiceTool.__new__(GovernmentServiceTool)
    tool.config = {"max_steps": 8}
    tool.instances = {}
    tool.completed = {}

    async def exercise():
        kwargs = {
            "create_kwargs": {
                "case": case.model_dump_json(),
                "_agent_request_id": "rollout-1",
            }
        }
        instance_id, _ = await tool.create(**kwargs)
        missing_slot = next(
            slot for slot in case.rules.required_slots if slot not in case.visible_slots
        )
        await tool.execute(
            instance_id,
            {"action": "ASK_USER", "slot": missing_slot},
        )
        await tool.release(instance_id)
        reused_id, _ = await tool.create(**kwargs)
        return instance_id, reused_id, missing_slot

    instance_id, reused_id, missing_slot = asyncio.run(exercise())
    assert instance_id == reused_id == "rollout-1"
    assert tool.instances[instance_id].known_slots[missing_slot] == case.hidden_truth[missing_slot]

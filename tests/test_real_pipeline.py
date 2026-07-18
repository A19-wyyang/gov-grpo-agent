from __future__ import annotations

import asyncio
import json

from gov_agent_rl.agent_env import GovernmentServiceEpisode
from gov_agent_rl.data_builder import (
    SCENARIO_COUNTS,
    build_cases,
    build_sft_messages,
    to_verl_row,
)
from gov_agent_rl.rewarding import score_episode
from gov_agent_rl.schema import ActionName
from gov_agent_rl.verl_reward import compute_score
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


def test_reward_hacking_and_unsafe_submit_are_hard_gated():
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
    score = score_episode(episode, expression_score=1.0)
    assert score.hard_gate
    assert score.total <= 0.2
    assert score.metrics["unsafe_submit"] == 1.0


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


def test_verl_reward_exports_replayed_environment_metrics():
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
    assert score["required_tool_rate"] == 1.0
    assert score["final_action_correct"] == 1.0


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

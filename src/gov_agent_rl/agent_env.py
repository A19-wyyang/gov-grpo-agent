from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .schema import ActionName, CaseSpec, EpisodeStep, StructuredAction

VERIFICATION_ORDER = (
    ActionName.POLICY_SEARCH,
    ActionName.ELIGIBILITY_CHECK,
    ActionName.MATERIAL_CHECK,
    ActionName.RISK_CHECK,
)


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == ">=":
        return actual is not None and actual >= expected
    if operator == "<=":
        return actual is not None and actual <= expected
    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    raise ValueError(f"unsupported eligibility operator: {operator}")


class GovernmentServiceEpisode:
    """Stateful, deterministic environment used by local tests and veRL tools."""

    def __init__(self, case: CaseSpec, max_steps: int = 8):
        self.case = case
        self.max_steps = max_steps
        self.known_slots = dict(case.visible_slots)
        self.tool_history: list[ActionName] = []
        self.tool_results: dict[str, Any] = {}
        self.asked_slots: list[str] = []
        self.failure_tags: list[str] = []
        self.failure_counts: Counter[str] = Counter()
        self.steps: list[EpisodeStep] = []
        self.attempt_history: list[dict[str, Any]] = []
        self.trailing_actions: list[dict[str, Any]] = []
        self.action_attempts = 0
        self.final_action: ActionName | None = None
        self.final_message = ""
        self.done = False

    @property
    def missing_slots(self) -> list[str]:
        return [slot for slot in self.case.rules.required_slots if slot not in self.known_slots]

    def initial_observation(self) -> dict[str, Any]:
        view = self.case.policy_view()
        return {
            "case": view.model_dump(mode="json"),
            "instruction": "请使用工具完成办理。标准答案和环境隐藏真值不会暴露给你。",
        }

    def execute(self, action_input: StructuredAction | dict[str, Any]) -> dict[str, Any]:
        if self.done:
            self.trailing_actions.append(self._serialize_action(action_input))
            self._failure("action_after_done")
            return {"ok": False, "error": "episode already finished"}
        if self.action_attempts >= self.max_steps:
            self._failure("max_steps_exceeded")
            self.done = True
            self.final_action = ActionName.REFUSE
            self.final_message = "办理轮次已超限，转人工处理。"
            return {"ok": False, "done": True, "error": "max steps exceeded"}

        self.action_attempts += 1
        raw_action = self._serialize_action(action_input)
        self.attempt_history.append(raw_action)
        try:
            action = (
                action_input
                if isinstance(action_input, StructuredAction)
                else StructuredAction.model_validate(action_input)
            )
        except Exception as exc:
            self._failure("illegal_action")
            return self._record_invalid(action_input, str(exc))

        observation = self._dispatch(action)
        self.steps.append(
            EpisodeStep(
                step=self.action_attempts,
                action=action,
                observation=observation,
                known_slots=dict(self.known_slots),
                tool_history=list(self.tool_history),
                failure_tags=list(self.failure_tags),
            )
        )
        return observation

    def _dispatch(self, action: StructuredAction) -> dict[str, Any]:
        if action.action == ActionName.ASK_USER:
            return self._ask_user(action.slot or "")
        if action.action == ActionName.POLICY_SEARCH:
            return self._policy_search(action.query or "")
        if action.action == ActionName.ELIGIBILITY_CHECK:
            return self._eligibility_check()
        if action.action == ActionName.MATERIAL_CHECK:
            return self._material_check()
        if action.action == ActionName.RISK_CHECK:
            return self._risk_check()
        return self._finish(action)

    def _ask_user(self, slot: str) -> dict[str, Any]:
        if slot in self.asked_slots:
            self._failure("repeated_question")
        self.asked_slots.append(slot)
        if slot not in self.case.rules.required_slots:
            self._failure("invalid_slot_question")
            return {"ok": False, "error": "该字段不属于本事项需要的信息"}
        if slot not in self.case.hidden_truth:
            self._failure("unknown_user_information")
            return {"ok": True, "slot": slot, "value": None, "message": "用户无法提供该信息"}
        value = self.case.hidden_truth[slot]
        self.known_slots[slot] = value
        return {"ok": True, "slot": slot, "value": value}

    def _policy_search(self, query: str) -> dict[str, Any]:
        self._register_tool(ActionName.POLICY_SEARCH)
        result = {
            "matter_id": self.case.matter_id,
            "title": self.case.title,
            "required_slots": self.case.rules.required_slots,
            "required_materials": self.case.rules.required_materials,
            "eligibility_rules": [rule.model_dump(mode="json") for rule in self.case.rules.eligibility_rules],
            "source": self.case.source.model_dump(mode="json"),
            "query": query,
        }
        self.tool_results[ActionName.POLICY_SEARCH.value] = result
        return {"ok": True, **result}

    def _eligibility_check(self) -> dict[str, Any]:
        self._register_tool(ActionName.ELIGIBILITY_CHECK)
        reasons: list[str] = []
        for slot in self.missing_slots:
            reasons.append(f"缺少必要信息：{slot}")
        for rule in self.case.rules.eligibility_rules:
            if not _compare(self.known_slots.get(rule.slot), rule.operator, rule.value):
                reasons.append(rule.failure_reason)
        result = {"eligible": not reasons, "reasons": reasons, "missing_slots": self.missing_slots}
        self.tool_results[ActionName.ELIGIBILITY_CHECK.value] = result
        return {"ok": True, **result}

    def _material_check(self) -> dict[str, Any]:
        self._register_tool(ActionName.MATERIAL_CHECK)
        materials = dict(self.case.hidden_truth.get("materials", {}))
        missing = [name for name in self.case.rules.required_materials if not materials.get(name)]
        result = {"complete": not missing, "missing_materials": missing}
        self.tool_results[ActionName.MATERIAL_CHECK.value] = result
        return {"ok": True, **result}

    def _risk_check(self) -> dict[str, Any]:
        self._register_tool(ActionName.RISK_CHECK)
        truth = self.case.hidden_truth.get("risk_flags", [])
        active = [flag for flag in self.case.rules.risk_flags if flag in truth]
        result = {"passed": not active, "risk_flags": active, "risk_level": "high" if active else "low"}
        self.tool_results[ActionName.RISK_CHECK.value] = result
        return {"ok": True, **result}

    def _finish(self, action: StructuredAction) -> dict[str, Any]:
        if action.action == ActionName.SUBMIT:
            if self.missing_slots:
                self._failure("premature_submit")
            missing_tools = [
                item.value for item in self.case.rules.required_tools if item not in self.tool_history
            ]
            if missing_tools:
                self._failure("missing_required_tool")
        self.final_action = action.action
        self.final_message = action.message or ""
        self.done = True
        return {
            "ok": True,
            "done": True,
            "final_action": action.action.value,
            "message": self.final_message,
        }

    def _record_invalid(self, action: Any, error: str) -> dict[str, Any]:
        observation = {"ok": False, "error": error}
        # Invalid actions count toward the horizon but cannot terminate the episode.
        return observation

    def _register_tool(self, action: ActionName) -> None:
        if action in self.tool_history:
            self._failure("repeated_tool_call")
        position = VERIFICATION_ORDER.index(action)
        if any(
            predecessor not in self.tool_history
            for predecessor in VERIFICATION_ORDER[:position]
        ):
            self._failure("tool_order_violation")
        if action == ActionName.ELIGIBILITY_CHECK and self.missing_slots:
            self._failure("eligibility_before_slots_complete")
        self.tool_history.append(action)

    @staticmethod
    def _serialize_action(action_input: Any) -> dict[str, Any]:
        if isinstance(action_input, StructuredAction):
            return action_input.model_dump(mode="json")
        if isinstance(action_input, dict):
            return dict(action_input)
        return {"raw_type": type(action_input).__name__}

    def _failure(self, tag: str) -> None:
        self.failure_counts[tag] += 1
        if tag not in self.failure_tags:
            self.failure_tags.append(tag)

    def trajectory(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "matter_id": self.case.matter_id,
            "steps": [step.model_dump(mode="json") for step in self.steps],
            "attempts": [*self.attempt_history, *self.trailing_actions],
            "action_attempts": self.action_attempts,
            "trailing_action_count": len(self.trailing_actions),
            "tool_history": [item.value for item in self.tool_history],
            "failure_tags": list(self.failure_tags),
            "final_action": self.final_action.value if self.final_action else None,
            "final_message": self.final_message,
            "done": self.done,
        }

    def render(self, observation: dict[str, Any]) -> str:
        return json.dumps(observation, ensure_ascii=False, sort_keys=True)

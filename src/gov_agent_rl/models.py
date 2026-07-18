from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal[
    "ASK_USER",
    "POLICY_SEARCH",
    "ELIGIBILITY_CHECK",
    "MATERIAL_CHECK",
    "RISK_CHECK",
    "SUBMIT",
    "REFUSE",
]


@dataclass(frozen=True)
class Action:
    type: ActionType
    slot: str | None = None
    query: str | None = None
    reason: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": self.type,
            "slot": self.slot,
            "query": self.query,
            "reason": self.reason,
            "message": self.message,
        }
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class GovCase:
    case_id: str
    title: str
    visible: dict[str, Any]
    hidden_truth: dict[str, Any]
    policy_rules: dict[str, Any]
    risk_rules: dict[str, Any]
    expected_result: dict[str, Any]

    @property
    def required_slots(self) -> list[str]:
        return list(self.policy_rules.get("required_slots", []))

    @property
    def required_tools(self) -> list[str]:
        return list(self.policy_rules.get("required_tools", []))


@dataclass
class RuntimeState:
    known_slots: dict[str, Any]
    tool_history: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    asked_slots: list[str] = field(default_factory=list)
    failure_tags: list[str] = field(default_factory=list)
    final_decision: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "known_slots": dict(self.known_slots),
            "tool_history": list(self.tool_history),
            "tool_results": dict(self.tool_results),
            "asked_slots": list(self.asked_slots),
        }


@dataclass
class TrajectoryStep:
    step: int
    state: dict[str, Any]
    action: dict[str, Any]
    observation: dict[str, Any]
    slot_status: dict[str, Any]
    failure_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "state": self.state,
            "action": self.action,
            "observation": self.observation,
            "slot_status": self.slot_status,
            "failure_tags": self.failure_tags,
        }


@dataclass
class Trajectory:
    case_id: str
    trajectory_id: str
    policy_name: str
    steps: list[TrajectoryStep]
    final_decision: dict[str, Any]
    failure_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "trajectory_id": self.trajectory_id,
            "policy_name": self.policy_name,
            "steps": [step.to_dict() for step in self.steps],
            "final_decision": self.final_decision,
            "failure_tags": self.failure_tags,
        }

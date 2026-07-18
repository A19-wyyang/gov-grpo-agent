from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionName(StrEnum):
    ASK_USER = "ASK_USER"
    POLICY_SEARCH = "POLICY_SEARCH"
    ELIGIBILITY_CHECK = "ELIGIBILITY_CHECK"
    MATERIAL_CHECK = "MATERIAL_CHECK"
    RISK_CHECK = "RISK_CHECK"
    SUBMIT = "SUBMIT"
    REFUSE = "REFUSE"


class SourceRef(BaseModel):
    url: str
    title: str
    authority: str
    region: str = "杭州市"
    retrieved_at: str
    content_sha256: str


class EligibilityRule(BaseModel):
    slot: str
    operator: str
    value: Any
    failure_reason: str


class MatterRules(BaseModel):
    required_slots: list[str]
    required_materials: list[str]
    required_tools: list[ActionName] = Field(
        default_factory=lambda: [
            ActionName.POLICY_SEARCH,
            ActionName.ELIGIBILITY_CHECK,
            ActionName.MATERIAL_CHECK,
            ActionName.RISK_CHECK,
        ]
    )
    eligibility_rules: list[EligibilityRule] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class ExpectedResult(BaseModel):
    final_action: ActionName
    reason_code: str
    reason: str


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    matter_id: str
    domain: str
    title: str
    split: str
    template_id: str
    scenario_type: str
    source: SourceRef
    user_request: str
    visible_slots: dict[str, Any]
    hidden_truth: dict[str, Any]
    rules: MatterRules
    expected_result: ExpectedResult
    reference_actions: list[ActionName]

    @model_validator(mode="after")
    def validate_reference(self) -> "CaseSpec":
        if not self.reference_actions:
            raise ValueError("reference_actions must not be empty")
        if self.reference_actions[-1] != self.expected_result.final_action:
            raise ValueError("reference path must end with the expected final action")
        return self

    def policy_view(self) -> "PolicyView":
        return PolicyView(
            case_id=self.case_id,
            matter_id=self.matter_id,
            title=self.title,
            user_request=self.user_request,
            known_slots=dict(self.visible_slots),
        )


class PolicyView(BaseModel):
    """The only case object that may be passed to a policy/model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    matter_id: str
    title: str
    user_request: str
    known_slots: dict[str, Any]


class StructuredAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionName
    slot: str | None = None
    query: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "StructuredAction":
        if self.action == ActionName.ASK_USER and not self.slot:
            raise ValueError("ASK_USER requires slot")
        if self.action == ActionName.POLICY_SEARCH and not self.query:
            raise ValueError("POLICY_SEARCH requires query")
        if self.action in {ActionName.SUBMIT, ActionName.REFUSE} and not self.message:
            raise ValueError(f"{self.action} requires message")
        return self


class EpisodeStep(BaseModel):
    step: int
    action: StructuredAction
    observation: dict[str, Any]
    known_slots: dict[str, Any]
    tool_history: list[ActionName]
    failure_tags: list[str]


class RewardBreakdown(BaseModel):
    hard_fact: float
    process: float
    expression: float | None = None
    penalties: dict[str, float] = Field(default_factory=dict)
    hard_gate: bool = False
    total: float
    metrics: dict[str, float] = Field(default_factory=dict)

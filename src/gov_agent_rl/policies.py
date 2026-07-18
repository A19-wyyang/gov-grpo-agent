from __future__ import annotations

from collections.abc import Callable

from .models import Action, GovCase, RuntimeState


Policy = Callable[[GovCase, RuntimeState, int], Action]


def careful_policy(case: GovCase, state: RuntimeState, step: int) -> Action:
    for slot in case.required_slots:
        if slot not in state.known_slots:
            return Action("ASK_USER", slot=slot)

    if "POLICY_SEARCH" not in state.tool_history:
        return Action("POLICY_SEARCH", query=case.visible.get("user_request", ""))
    if "ELIGIBILITY_CHECK" not in state.tool_history:
        return Action("ELIGIBILITY_CHECK")
    if "MATERIAL_CHECK" not in state.tool_history:
        return Action("MATERIAL_CHECK")
    if "RISK_CHECK" not in state.tool_history:
        return Action("RISK_CHECK")

    eligibility = state.tool_results["ELIGIBILITY_CHECK"]
    materials = state.tool_results["MATERIAL_CHECK"]
    risks = state.tool_results["RISK_CHECK"]
    if not eligibility["eligible"]:
        return Action("REFUSE", message=f"暂不能提交：{'；'.join(eligibility['reasons'])}。")
    if materials["missing_materials"]:
        missing = "、".join(materials["missing_materials"])
        return Action("REFUSE", message=f"暂不能提交：缺少材料 {missing}，请补齐后再次申请。")
    if risks["risk_flags"]:
        return Action("REFUSE", message="暂不能提交：风险核验未通过，建议转人工处理。")
    return Action("SUBMIT", message="经核验，您符合办理条件，材料齐全，已提交申请。")


def risky_policy(case: GovCase, state: RuntimeState, step: int) -> Action:
    if step == 1:
        return Action("POLICY_SEARCH", query=case.visible.get("user_request", ""))
    if step == 2:
        return Action("SUBMIT", message="看起来符合条件，先为您提交申请。")
    return Action("SUBMIT", message="已提交。")


def over_refuse_policy(case: GovCase, state: RuntimeState, step: int) -> Action:
    missing_slots = [slot for slot in case.required_slots if slot not in state.known_slots]
    if step == 1 and missing_slots:
        return Action("ASK_USER", slot=missing_slots[0])
    if step == 2 and state.asked_slots:
        return Action("ASK_USER", slot=state.asked_slots[0])
    return Action("REFUSE", message="当前信息不足，建议暂不办理。")


def conservative_refuse_policy(case: GovCase, state: RuntimeState, step: int) -> Action:
    """A collapse mode: polished refusal without completing the workflow."""
    return Action(
        "REFUSE",
        message="为避免错误提交，当前暂不能办理。建议补齐材料并确认资格后再次申请。",
    )


def judge_hacker_policy(case: GovCase, state: RuntimeState, step: int) -> Action:
    """A reward-hacking mode: looks helpful but intentionally skips key tools."""
    for slot in case.required_slots:
        if slot not in state.known_slots:
            return Action("ASK_USER", slot=slot)
    if "POLICY_SEARCH" not in state.tool_history:
        return Action("POLICY_SEARCH", query=case.visible.get("user_request", ""))
    if "ELIGIBILITY_CHECK" not in state.tool_history:
        return Action("ELIGIBILITY_CHECK")
    return Action(
        "REFUSE",
        message="资格已初步核验。为确保顺利办理，建议补齐相关材料后再次提交申请。",
    )


POLICIES: dict[str, Policy] = {
    "careful_policy": careful_policy,
    "risky_policy": risky_policy,
    "over_refuse_policy": over_refuse_policy,
    "conservative_refuse_policy": conservative_refuse_policy,
    "judge_hacker_policy": judge_hacker_policy,
}

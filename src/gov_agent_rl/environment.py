from __future__ import annotations

from typing import Any

from .models import Action, GovCase, RuntimeState


TOOL_ACTIONS = {
    "POLICY_SEARCH",
    "ELIGIBILITY_CHECK",
    "MATERIAL_CHECK",
    "RISK_CHECK",
}


class GovernmentServiceEnv:
    def __init__(self, case: GovCase, max_steps: int = 8):
        self.case = case
        self.max_steps = max_steps
        self.state = RuntimeState(
            known_slots=dict(case.visible.get("known_slots", {})),
        )
        self.step_count = 0
        self.done = False

    def step(self, action: Action) -> dict[str, Any]:
        if self.done:
            self._add_failure("action_after_done")
            return {"error": "trajectory already finished"}

        self.step_count += 1
        observation = self._execute(action)

        if self.step_count >= self.max_steps and not self.done:
            self._add_failure("max_steps_exceeded")
            self.done = True
            self.state.final_decision = {
                "type": "REFUSE",
                "message": "办理轮次已超限，当前信息不足，建议转人工处理。",
            }
            observation["auto_stop"] = True

        return observation

    def slot_status(self) -> dict[str, Any]:
        filled = sorted(
            slot for slot in self.case.required_slots if slot in self.state.known_slots
        )
        missing = sorted(
            slot for slot in self.case.required_slots if slot not in self.state.known_slots
        )
        return {"filled": filled, "missing": missing}

    def _execute(self, action: Action) -> dict[str, Any]:
        if action.type == "ASK_USER":
            return self._ask_user(action)
        if action.type == "POLICY_SEARCH":
            return self._policy_search(action)
        if action.type == "ELIGIBILITY_CHECK":
            return self._eligibility_check()
        if action.type == "MATERIAL_CHECK":
            return self._material_check()
        if action.type == "RISK_CHECK":
            return self._risk_check()
        if action.type in {"SUBMIT", "REFUSE"}:
            return self._finish(action)
        self._add_failure("illegal_action")
        return {"error": f"unsupported action: {action.type}"}

    def _ask_user(self, action: Action) -> dict[str, Any]:
        if not action.slot:
            self._add_failure("missing_action_parameter")
            return {"error": "ASK_USER requires slot"}
        if action.slot in self.state.asked_slots:
            self._add_failure("repeated_question")
        self.state.asked_slots.append(action.slot)

        if action.slot in self.case.hidden_truth:
            value = self.case.hidden_truth[action.slot]
            self.state.known_slots[action.slot] = value
            return {action.slot: value}

        self._add_failure("invalid_slot_question")
        return {"error": f"slot {action.slot} not found in user truth"}

    def _policy_search(self, action: Action) -> dict[str, Any]:
        if not action.query:
            self._add_failure("missing_action_parameter")
        self.state.tool_history.append("POLICY_SEARCH")
        result = {
            "policy_rules": self.case.policy_rules,
            "matched_query": action.query or self.case.visible.get("user_request"),
        }
        self.state.tool_results["POLICY_SEARCH"] = result
        return result

    def _eligibility_check(self) -> dict[str, Any]:
        self.state.tool_history.append("ELIGIBILITY_CHECK")
        required_slots = self.case.required_slots
        missing = [slot for slot in required_slots if slot not in self.state.known_slots]
        min_months = self.case.policy_rules.get("min_social_security_months")
        months = self.state.known_slots.get("social_security_months")
        eligible = not missing
        reasons = []

        if missing:
            eligible = False
            reasons.append(f"缺少必要槽位: {', '.join(missing)}")
        if min_months is not None and months is not None and months < min_months:
            eligible = False
            reasons.append(f"社保缴纳月数不足 {min_months} 个月")

        result = {
            "eligible": eligible,
            "missing_slots": missing,
            "reasons": reasons,
        }
        self.state.tool_results["ELIGIBILITY_CHECK"] = result
        return result

    def _material_check(self) -> dict[str, Any]:
        self.state.tool_history.append("MATERIAL_CHECK")
        materials = self.case.hidden_truth.get("materials", {})
        required = self.case.policy_rules.get("required_materials", [])
        missing = [name for name in required if not materials.get(name)]
        result = {
            "materials": dict(materials),
            "missing_materials": missing,
            "complete": not missing,
        }
        self.state.tool_results["MATERIAL_CHECK"] = result
        return result

    def _risk_check(self) -> dict[str, Any]:
        self.state.tool_history.append("RISK_CHECK")
        configured_risks = self.case.risk_rules.get("risk_flags", [])
        truth_risks = self.case.hidden_truth.get("risk_flags", [])
        active = [flag for flag in configured_risks if flag in truth_risks]
        result = {"risk_flags": active, "risk_level": "high" if active else "low"}
        self.state.tool_results["RISK_CHECK"] = result
        return result

    def _finish(self, action: Action) -> dict[str, Any]:
        missing = self.slot_status()["missing"]
        if action.type == "SUBMIT" and missing:
            self._add_failure("early_submit")

        self.done = True
        self.state.final_decision = {
            "type": action.type,
            "message": action.message or action.reason or "",
        }
        return {"final_decision": self.state.final_decision}

    def _add_failure(self, tag: str) -> None:
        self.state.failure_tags.append(tag)

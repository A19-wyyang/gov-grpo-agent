VERIFIER_WEIGHTS = {
    "slot_completion": 0.15,
    "policy_search": 0.10,
    "eligibility_check": 0.15,
    "material_check": 0.15,
    "final_decision": 0.20,
    "missing_materials": 0.10,
    "valid_actions": 0.10,
    "no_premature_submit": 0.05,
}


def score_trajectory(case, trajectory):
    steps = trajectory.get("steps", [])
    called = [step.get("action") for step in steps]
    failure_reasons = []
    verifier_score = 0.0

    if not case["hidden_truth"]["missing_slots"] or "Ask_User" in called:
        verifier_score += VERIFIER_WEIGHTS["slot_completion"]
    else:
        failure_reasons.append("missing_slots:not_asked")

    tool_to_weight = {
        "Policy_Search": "policy_search",
        "Eligibility_Check": "eligibility_check",
        "Material_Check": "material_check",
    }
    for tool_name, weight_name in tool_to_weight.items():
        if tool_name in called:
            verifier_score += VERIFIER_WEIGHTS[weight_name]

    if trajectory.get("final_answer") == case["hidden_truth"]["final_decision"]:
        verifier_score += VERIFIER_WEIGHTS["final_decision"]
    else:
        failure_reasons.append("final_decision:mismatch")

    material_observation = _last_observation(steps, "Material_Check")
    if material_observation is not None and material_observation.get("missing", []) == case["hidden_truth"]["missing_materials"]:
        verifier_score += VERIFIER_WEIGHTS["missing_materials"]
    elif not case["hidden_truth"]["missing_materials"]:
        verifier_score += VERIFIER_WEIGHTS["missing_materials"]
    else:
        failure_reasons.append("missing_materials:mismatch")

    if all(step.get("action") for step in steps):
        verifier_score += VERIFIER_WEIGHTS["valid_actions"]

    if not _is_premature_submit(case, called):
        verifier_score += VERIFIER_WEIGHTS["no_premature_submit"]

    penalty = 0.0
    for required_tool in case["hidden_truth"]["required_tools"]:
        if required_tool not in called:
            penalty += 0.25 if required_tool in {"Eligibility_Check", "Material_Check"} else 0.15
            failure_reasons.append(f"missing_required_tool:{required_tool}")

    if _is_premature_submit(case, called):
        penalty += 0.30
        failure_reasons.append("premature_submit:missing_slots")

    judge_score = _judge_final_answer(trajectory.get("final_answer", ""))
    reward = max(0.0, min(1.0, 0.85 * verifier_score + 0.15 * judge_score - penalty))
    return {
        "case_id": case["case_id"],
        "rollout_id": trajectory.get("rollout_id"),
        "verifier_score": round(verifier_score, 4),
        "judge_score": round(judge_score, 4),
        "penalty": round(penalty, 4),
        "reward": round(reward, 4),
        "failure_reasons": failure_reasons,
    }


def _last_observation(steps, action_name):
    for step in reversed(steps):
        if step.get("action") == action_name:
            return step.get("observation", {})
    return None


def _is_premature_submit(case, called):
    if "Submit" not in called and "Refuse" not in called:
        return False
    if case["hidden_truth"]["missing_slots"] and "Ask_User" not in called:
        return True
    return False


def _judge_final_answer(final_answer):
    if not final_answer:
        return 0.0
    completeness = 1.0 if any(word in final_answer for word in ["符合", "不符合", "材料", "补充"]) else 0.6
    clarity = 1.0 if len(final_answer) >= 10 else 0.5
    actionability = 1.0 if any(word in final_answer for word in ["提交", "申请", "补充", "暂不能"]) else 0.6
    return 0.4 * completeness + 0.3 * clarity + 0.3 * actionability

def infer_decision_label(final_answer):
    if not isinstance(final_answer, str) or not final_answer.strip():
        return "unknown"
    text = final_answer.strip()
    if any(token in text for token in ["信息不完整", "补充办理城市", "补充信息", "先补充"]):
        return "missing_information"
    if any(token in text for token in ["材料不完整", "缺少", "补充"]) and "材料齐全" not in text:
        return "material_missing"
    if any(token in text for token in ["不符合", "暂不能", "不能提交", "无法办理"]):
        return "ineligible"
    if any(token in text for token in ["符合", "材料齐全", "可提交", "可以办理", "可申请"]):
        return "success"
    return "unknown"


def expected_decision_label(case):
    hidden = case["hidden_truth"]
    if hidden["missing_slots"]:
        return "missing_information"
    if not hidden["eligible"]:
        return "ineligible"
    if hidden["missing_materials"]:
        return "material_missing"
    return "success"


def evaluate_final_decision(case, final_answer):
    expected = expected_decision_label(case)
    actual = infer_decision_label(final_answer)
    failures = []
    if not isinstance(final_answer, str):
        failures.append("final_answer:not_string")
    if actual != expected:
        failures.append(f"decision_label:mismatch:{actual}!={expected}")
    if expected == "material_missing" and isinstance(final_answer, str):
        for material in case["hidden_truth"]["missing_materials"]:
            if material not in final_answer:
                failures.append(f"missing_material:not_mentioned:{material}")
    correct = not failures
    return {
        "correct": correct,
        "expected_label": expected,
        "actual_label": actual,
        "failure_reasons": failures,
    }

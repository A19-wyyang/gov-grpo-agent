import json


SAMPLE_TYPE_WEIGHTS = [
    ("single_action", 0.30),
    ("multi_tool", 0.30),
    ("missing_slot_question", 0.15),
    ("material_missing_answer", 0.10),
    ("ineligible_refusal", 0.10),
    ("complex_multiturn", 0.05),
]


def build_sft_samples(cases, target_count=2000):
    samples = []
    case_list = list(cases)
    sample_types = _planned_sample_types(target_count)
    for index, sample_type in enumerate(sample_types):
        case = case_list[index % len(case_list)]
        action = _action_for_sample(case, sample_type)
        samples.append(
            {
                "sample_id": f"sft_{index + 1:05d}",
                "case_id": case["case_id"],
                "sample_type": sample_type,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是政务办理Agent。必须输出合法JSON动作，不得跳过必要工具核验。",
                    },
                    {
                        "role": "user",
                        "content": _user_prompt(case, sample_type),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    },
                ],
            }
        )
    return samples


def _planned_sample_types(target_count):
    sample_types = []
    assigned = 0
    for sample_type, weight in SAMPLE_TYPE_WEIGHTS[:-1]:
        count = int(target_count * weight)
        sample_types.extend([sample_type] * count)
        assigned += count
    last_type = SAMPLE_TYPE_WEIGHTS[-1][0]
    sample_types.extend([last_type] * (target_count - assigned))
    return sample_types


def _action_for_sample(case, sample_type):
    if sample_type == "missing_slot_question" or (
        sample_type == "single_action" and case["hidden_truth"]["missing_slots"]
    ):
        return {
            "action": "Ask_User",
            "arguments": {"slots": case["hidden_truth"]["missing_slots"] or ["city"]},
        }
    if sample_type == "material_missing_answer":
        return {
            "action": "Submit",
            "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
        }
    if sample_type == "ineligible_refusal":
        return {
            "action": "Refuse",
            "arguments": {"final_answer": case["hidden_truth"]["final_decision"]},
        }
    if sample_type == "multi_tool":
        return {
            "action": "Policy_Search",
            "arguments": {
                "service_item": case["service_item"],
                "city": case["user_profile"].get("city"),
                "query": "申请条件 材料",
            },
        }
    if sample_type == "complex_multiturn":
        return {"action": "Eligibility_Check", "arguments": {}}
    return {
        "action": "Policy_Search",
        "arguments": {
            "service_item": case["service_item"],
            "city": case["user_profile"].get("city"),
            "query": "申请条件 材料",
        },
    }


def _user_prompt(case, sample_type):
    return (
        f"样本类型：{sample_type}\n"
        f"事项：{case['service_item']}\n"
        f"用户诉求：{case['user_initial_query']}\n"
        f"路径类型：{case['path_type']}\n"
        "请给出下一步结构化动作。"
    )

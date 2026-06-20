def compute_metrics(cases, trajectories, reward_reports):
    case_by_id = {case["case_id"]: case for case in cases}
    report_by_rollout = {report["rollout_id"]: report for report in reward_reports}
    total = len(trajectories) or 1
    successes = 0
    required_tools = 0
    called_required_tools = 0
    premature = 0
    missing_tool_trajectories = 0
    material_check_calls = 0
    final_decision_correct = 0
    invalid_actions = 0

    for trajectory in trajectories:
        case = case_by_id[trajectory["case_id"]]
        called = [step["action"] for step in trajectory.get("steps", [])]
        required = case["hidden_truth"]["required_tools"]
        required_tools += len(required)
        called_required_tools += sum(1 for tool in required if tool in called)
        if any(tool not in called for tool in required):
            missing_tool_trajectories += 1
        if "Material_Check" in called:
            material_check_calls += 1
        if trajectory.get("final_answer") == case["hidden_truth"]["final_decision"]:
            final_decision_correct += 1
        if case["hidden_truth"]["missing_slots"] and "Ask_User" not in called and (
            "Submit" in called or "Refuse" in called
        ):
            premature += 1
        invalid_actions += sum(1 for action in called if not action)
        report = report_by_rollout.get(trajectory.get("rollout_id"), {})
        if report.get("reward", 0) >= 0.8:
            successes += 1

    return {
        "success_at_1": round(successes / total, 4),
        "required_tool_recall": round(called_required_tools / (required_tools or 1), 4),
        "premature_submit_rate": round(premature / total, 4),
        "missing_tool_rate": round(missing_tool_trajectories / total, 4),
        "material_check_call_rate": round(material_check_calls / total, 4),
        "final_decision_accuracy": round(final_decision_correct / total, 4),
        "invalid_action_rate": round(invalid_actions / total, 4),
    }

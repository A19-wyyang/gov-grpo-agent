ALLOWED_ACTIONS = {
    "Ask_User",
    "Policy_Search",
    "Eligibility_Check",
    "Material_Check",
    "Risk_Check",
    "Submit",
    "Refuse",
}


class ActionValidationError(ValueError):
    """Raised when a model action violates the action contract."""


def validate_case(case):
    required_top_level = {
        "case_id",
        "domain",
        "service_item",
        "user_initial_query",
        "user_profile",
        "hidden_truth",
        "difficulty",
        "error_type",
        "path_type",
    }
    hidden_required = {
        "eligible",
        "missing_slots",
        "required_tools",
        "required_materials",
        "missing_materials",
        "final_decision",
    }
    return required_top_level.issubset(case) and hidden_required.issubset(case["hidden_truth"])


def validate_action(action):
    if not isinstance(action, dict):
        raise ActionValidationError("action must be a dict")
    if action.get("action") not in ALLOWED_ACTIONS:
        raise ActionValidationError(f"unknown action: {action.get('action')}")
    if "arguments" not in action or not isinstance(action["arguments"], dict):
        raise ActionValidationError("action.arguments must be a dict")
    return True


def validate_trajectory(trajectory):
    required = {"case_id", "rollout_id", "steps", "final_answer", "metadata"}
    if not required.issubset(trajectory):
        return False
    if not isinstance(trajectory["steps"], list) or not trajectory["steps"]:
        return False
    for step in trajectory["steps"]:
        if not {"turn", "action", "arguments", "observation"}.issubset(step):
            return False
        validate_action({"action": step["action"], "arguments": step["arguments"]})
    return True

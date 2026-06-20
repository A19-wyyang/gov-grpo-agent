from gov_grpo_agent.data import load_policy_catalog


class PolicySearch:
    name = "Policy_Search"

    def run(self, arguments, case):
        catalog = load_policy_catalog()
        service_item = arguments.get("service_item") or case["service_item"]
        policy = catalog[service_item]
        return {
            "conditions": list(policy["conditions"]),
            "required_materials": list(policy["required_materials"]),
            "city": arguments.get("city") or case["user_profile"].get("city") or policy["city"],
        }


class EligibilityCheck:
    name = "Eligibility_Check"

    def run(self, arguments, case):
        failed_conditions = []
        profile = case["user_profile"]
        if case["service_item"] == "租房提取公积金" and profile.get("continuous_payment_months", 0) < 6:
            failed_conditions.append("continuous_payment_months")
        if case["service_item"] == "医保异地备案" and not profile.get("insured", True):
            failed_conditions.append("insured")
        if case["service_item"] == "失业保险申领" and profile.get("insurance_months", 12) < 12:
            failed_conditions.append("insurance_months")
        if case["service_item"] == "人才补贴申请" and not profile.get("employed_local", True):
            failed_conditions.append("employed_local")
        if case["service_item"] == "个体工商户注册" and not profile.get("valid_location", True):
            failed_conditions.append("valid_location")

        uncertain_slots = [slot for slot in case["hidden_truth"]["missing_slots"] if slot != "city"]
        eligible = case["hidden_truth"]["eligible"] and not failed_conditions
        return {
            "eligible": eligible,
            "failed_conditions": failed_conditions,
            "uncertain_slots": uncertain_slots,
            "explanation": "满足基本资格条件。" if eligible else "存在不满足的资格条件。",
        }


class MaterialCheck:
    name = "Material_Check"

    def run(self, arguments, case):
        required = case["hidden_truth"]["required_materials"]
        missing = case["hidden_truth"]["missing_materials"]
        provided = [material for material in required if material not in missing]
        return {
            "complete": not missing,
            "provided": provided,
            "missing": list(missing),
            "explanation": "材料齐全。" if not missing else f"材料不完整，缺少：{', '.join(missing)}。",
        }


TOOL_REGISTRY = {
    "Policy_Search": PolicySearch(),
    "Eligibility_Check": EligibilityCheck(),
    "Material_Check": MaterialCheck(),
}

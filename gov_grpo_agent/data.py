from copy import deepcopy


SERVICE_DEFINITIONS = {
    "housing_fund": {
        "domain": "住房公积金",
        "service_item": "租房提取公积金",
        "city": "杭州",
        "query": "我想提取公积金交房租，应该怎么办？",
        "conditions": ["正常缴存住房公积金", "本市无自有住房", "需提供有效租赁材料"],
        "required_materials": ["身份证", "银行卡", "租赁合同", "租赁备案证明"],
        "tools": ["Policy_Search", "Eligibility_Check", "Material_Check"],
        "profile": {
            "city": "杭州",
            "employment_status": "在职",
            "has_housing_fund_account": True,
            "continuous_payment_months": 8,
            "has_own_house": False,
            "rental_contract": True,
            "id_card": True,
            "bank_card": True,
            "rental_filing": True,
        },
    },
    "medical_remote": {
        "domain": "社保医保",
        "service_item": "医保异地备案",
        "city": "杭州",
        "query": "我要去外地看病，医保怎么备案？",
        "conditions": ["已参加基本医保", "提供异地就医地信息", "备案原因真实有效"],
        "required_materials": ["身份证", "医保电子凭证", "异地就医地信息"],
        "tools": ["Policy_Search", "Eligibility_Check", "Material_Check"],
        "profile": {"city": "杭州", "insured": True, "target_city": "上海", "id_card": True, "medical_card": True},
    },
    "unemployment": {
        "domain": "社保医保",
        "service_item": "失业保险申领",
        "city": "杭州",
        "query": "我离职了，想领失业保险金。",
        "conditions": ["非本人意愿中断就业", "失业保险缴费满一年", "已办理失业登记"],
        "required_materials": ["身份证", "社保卡", "解除劳动关系证明", "失业登记信息"],
        "tools": ["Policy_Search", "Eligibility_Check", "Material_Check"],
        "profile": {"city": "杭州", "involuntary_unemployed": True, "insurance_months": 18, "registered_unemployed": True, "id_card": True},
    },
    "talent_subsidy": {
        "domain": "人才服务",
        "service_item": "人才补贴申请",
        "city": "杭州",
        "query": "我想申请人才补贴，需要什么条件？",
        "conditions": ["符合学历或职称要求", "在本市就业参保", "未重复享受同类补贴"],
        "required_materials": ["身份证", "学历证明", "劳动合同", "社保缴纳证明"],
        "tools": ["Policy_Search", "Eligibility_Check", "Material_Check"],
        "profile": {"city": "杭州", "degree": "本科", "employed_local": True, "social_security_months": 6, "id_card": True},
    },
    "business_license": {
        "domain": "市场监管",
        "service_item": "个体工商户注册",
        "city": "杭州",
        "query": "我要开一家小店，怎么注册个体工商户？",
        "conditions": ["经营者身份真实", "经营场所合法", "名称和经营范围符合规范"],
        "required_materials": ["身份证", "经营场所证明", "个体工商户登记申请书"],
        "tools": ["Policy_Search", "Eligibility_Check", "Material_Check"],
        "profile": {"city": "杭州", "id_card": True, "valid_location": True, "business_scope": "餐饮服务"},
    },
}


PATH_DISTRIBUTION = [
    ("simple_success", 60),
    ("missing_information", 50),
    ("ineligible", 40),
    ("material_missing", 30),
    ("complex_mixed", 20),
]


def load_policy_catalog():
    return {
        item["service_item"]: {
            "domain": item["domain"],
            "city": item["city"],
            "conditions": list(item["conditions"]),
            "required_materials": list(item["required_materials"]),
            "tools": list(item["tools"]),
        }
        for item in SERVICE_DEFINITIONS.values()
    }


def build_case(service_key, index, path_type):
    service = SERVICE_DEFINITIONS[service_key]
    profile = deepcopy(service["profile"])
    missing_slots = []
    missing_materials = []
    eligible = True
    difficulty = "easy"
    error_type = "none"
    final_decision = "符合办理条件，材料齐全，可按流程提交申请。"

    if path_type == "missing_information":
        profile["city"] = None
        missing_slots = ["city"]
        difficulty = "medium"
        error_type = "slot_missing"
        final_decision = "信息不完整，需先补充办理城市后再核验政策和材料。"
    elif path_type == "ineligible":
        eligible = False
        difficulty = "medium"
        error_type = "eligibility_failed"
        final_decision = "不符合当前办理条件，暂不能提交申请。"
        if service_key == "housing_fund":
            profile["continuous_payment_months"] = 2
        elif service_key == "medical_remote":
            profile["insured"] = False
        elif service_key == "unemployment":
            profile["insurance_months"] = 6
        elif service_key == "talent_subsidy":
            profile["employed_local"] = False
        else:
            profile["valid_location"] = False
    elif path_type == "material_missing":
        difficulty = "medium"
        error_type = "material_missing"
        missing_materials = [service["required_materials"][-1]]
        final_decision = f"材料不完整，需补充{missing_materials[0]}后申请。"
    elif path_type == "complex_mixed":
        difficulty = "hard"
        error_type = "complex_mixed"
        missing_slots = ["city"]
        missing_materials = [service["required_materials"][-1]]
        profile["city"] = None
        final_decision = f"需先补充办理城市，并补齐{missing_materials[0]}后再提交。"

    case_id = f"{service_key}_{index:04d}"
    return {
        "case_id": case_id,
        "domain": service["domain"],
        "service_item": service["service_item"],
        "user_initial_query": service["query"],
        "user_profile": profile,
        "hidden_truth": {
            "eligible": eligible,
            "missing_slots": missing_slots,
            "required_tools": list(service["tools"]),
            "required_materials": list(service["required_materials"]),
            "missing_materials": missing_materials,
            "final_decision": final_decision,
        },
        "difficulty": difficulty,
        "error_type": error_type,
        "path_type": path_type,
    }


def build_mvp_cases(limit=None):
    cases = []
    service_keys = list(SERVICE_DEFINITIONS)
    for path_type, count in PATH_DISTRIBUTION:
        for offset in range(count):
            service_key = service_keys[len(cases) % len(service_keys)]
            cases.append(build_case(service_key, len(cases) + 1, path_type))
    return cases if limit is None else cases[:limit]

from copy import deepcopy


SERVICE_ITEM_ALIASES = {
    "住房公积金提取": "租房提取公积金",
    "公积金租房提取": "租房提取公积金",
    "租房提取": "租房提取公积金",
    "租房提取公积金": "租房提取公积金",
    "异地医保备案": "医保异地备案",
    "医保异地就医备案": "医保异地备案",
    "领取失业保险": "失业保险申领",
    "失业金申领": "失业保险申领",
    "人才补贴": "人才补贴申请",
    "个体户注册": "个体工商户注册",
    "个体工商户登记": "个体工商户注册",
}


def normalize_service_item(service_item):
    if not isinstance(service_item, str):
        return service_item
    compact = service_item.strip()
    return SERVICE_ITEM_ALIASES.get(compact, compact)


def normalize_action_arguments(action):
    normalized = deepcopy(action)
    arguments = normalized.setdefault("arguments", {})
    if "service_item" in arguments:
        arguments["service_item"] = normalize_service_item(arguments["service_item"])
    return normalized

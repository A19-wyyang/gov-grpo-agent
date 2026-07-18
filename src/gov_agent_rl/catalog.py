from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatterTemplate:
    matter_id: str
    domain: str
    title: str
    request: str
    required_slots: tuple[str, ...]
    required_materials: tuple[str, ...]
    risk_flags: tuple[str, ...]
    source_url: str
    source_title: str
    authority: str
    base_truth: dict[str, Any]
    eligibility: tuple[dict[str, Any], ...] = ()


def _source_hash(title: str, url: str) -> str:
    return hashlib.sha256(f"{title}\n{url}".encode()).hexdigest()


# The catalog is a versioned seed for data generation. Before a formal run,
# source URLs must be reviewed and the manifest refreshed with archived content.
MATTERS: tuple[MatterTemplate, ...] = (
    MatterTemplate(
        "business_registration",
        "市场监管",
        "个体工商户设立登记",
        "我想在杭州开一家小餐饮店，怎么办营业执照？",
        ("city", "business_type", "applicant_type", "has_fixed_location"),
        ("id_card", "location_certificate", "application_form"),
        ("duplicate_application", "identity_mismatch"),
        "https://scjg.hangzhou.gov.cn/",
        "杭州市市场监督管理局网上办事",
        "杭州市市场监督管理局",
        {"city": "杭州", "business_type": "小餐饮", "applicant_type": "individual", "has_fixed_location": True},
    ),
    MatterTemplate(
        "residence_permit_endorsement",
        "户政",
        "居住证签注",
        "我的居住证快到期了，想办理签注。",
        ("city", "residence_months", "has_stable_address"),
        ("id_card", "residence_permit", "address_certificate"),
        ("identity_mismatch", "expired_too_long"),
        "https://www.gongshu.gov.cn/",
        "居住登记及居住证管理基层政务公开目录",
        "杭州市拱墅区人民政府",
        {"city": "杭州", "residence_months": 8, "has_stable_address": True},
        ({"slot": "residence_months", "operator": ">=", "value": 6, "failure_reason": "居住登记时长不足"},),
    ),
    MatterTemplate(
        "flexible_employment_subsidy",
        "社会保障",
        "灵活就业社会保险补贴",
        "我想申请灵活就业社保补贴。",
        ("city", "employment_status", "social_security_months"),
        ("id_card", "employment_certificate", "payment_record"),
        ("duplicate_application", "abnormal_social_security_record"),
        "https://hrss.hangzhou.gov.cn/",
        "杭州市人力资源和社会保障局办事服务",
        "杭州市人力资源和社会保障局",
        {"city": "杭州", "employment_status": "flexible_worker", "social_security_months": 14},
        ({"slot": "social_security_months", "operator": ">=", "value": 12, "failure_reason": "社保缴费月数不足"},),
    ),
    MatterTemplate(
        "graduate_living_subsidy",
        "社会保障",
        "应届毕业生生活补贴",
        "我是刚来杭州工作的应届毕业生，想申请生活补贴。",
        ("city", "degree", "graduation_months", "employed_in_hangzhou"),
        ("id_card", "degree_certificate", "social_security_record"),
        ("duplicate_application", "degree_verification_failed"),
        "https://hrss.hangzhou.gov.cn/art/2024/5/13/art_1229578386_4267025.html",
        "杭州市应届毕业生生活补贴申请指南",
        "杭州市人力资源和社会保障局",
        {"city": "杭州", "degree": "master", "graduation_months": 6, "employed_in_hangzhou": True},
        ({"slot": "graduation_months", "operator": "<=", "value": 24, "failure_reason": "毕业时间超出申请范围"},),
    ),
    MatterTemplate(
        "provident_fund_withdrawal_rent",
        "住房公积金",
        "租赁住房提取住房公积金",
        "我在杭州租房，想提取公积金交房租。",
        ("city", "has_local_house", "continuous_contribution_months"),
        ("id_card", "rental_record", "bank_card"),
        ("duplicate_withdrawal", "property_record_conflict"),
        "https://gjj.hangzhou.gov.cn/",
        "杭州住房公积金管理中心办事指南",
        "杭州住房公积金管理中心",
        {"city": "杭州", "has_local_house": False, "continuous_contribution_months": 6},
        ({"slot": "continuous_contribution_months", "operator": ">=", "value": 3, "failure_reason": "公积金连续缴存时间不足"},),
    ),
    MatterTemplate(
        "provident_fund_loan",
        "住房公积金",
        "住房公积金个人住房贷款",
        "我准备在杭州买首套房，想申请公积金贷款。",
        ("city", "continuous_contribution_months", "credit_status", "house_count"),
        ("id_card", "purchase_contract", "income_certificate"),
        ("credit_risk", "duplicate_loan"),
        "https://gjj.hangzhou.gov.cn/",
        "杭州住房公积金贷款办事指南",
        "杭州住房公积金管理中心",
        {"city": "杭州", "continuous_contribution_months": 18, "credit_status": "normal", "house_count": 0},
        ({"slot": "continuous_contribution_months", "operator": ">=", "value": 6, "failure_reason": "公积金连续缴存时间不足"},),
    ),
    MatterTemplate(
        "birth_registration",
        "户政",
        "出生登记",
        "孩子刚出生，想在杭州办理出生登记。",
        ("city", "parent_identity_confirmed", "birth_hospital"),
        ("parent_id_cards", "birth_medical_certificate", "household_book"),
        ("certificate_mismatch", "duplicate_registration"),
        "https://www.gongshu.gov.cn/",
        "出生登记基层政务公开目录",
        "杭州市公安局拱墅区分局",
        {"city": "杭州", "parent_identity_confirmed": True, "birth_hospital": "杭州某医院"},
    ),
    MatterTemplate(
        "household_migration",
        "户政",
        "户口迁移登记",
        "我想把户口迁到杭州。",
        ("target_city", "migration_reason", "has_legal_residence"),
        ("id_card", "household_book", "residence_certificate"),
        ("identity_mismatch", "false_residence"),
        "https://www.gongshu.gov.cn/",
        "迁移登记基层政务公开目录",
        "杭州市公安局拱墅区分局",
        {"target_city": "杭州", "migration_reason": "employment", "has_legal_residence": True},
    ),
    MatterTemplate(
        "food_business_license",
        "市场监管",
        "食品经营许可",
        "我准备在杭州开餐馆，想申请食品经营许可证。",
        ("city", "business_scope", "has_compliant_site", "food_safety_manager"),
        ("id_card", "site_plan", "safety_management_system"),
        ("site_risk", "responsible_person_blacklist"),
        "https://scjg.hangzhou.gov.cn/",
        "杭州市市场监督管理局食品经营许可服务",
        "杭州市市场监督管理局",
        {"city": "杭州", "business_scope": "restaurant", "has_compliant_site": True, "food_safety_manager": True},
    ),
    MatterTemplate(
        "company_change_registration",
        "市场监管",
        "公司变更登记",
        "公司法定代表人发生变化，如何办理变更登记？",
        ("city", "company_status", "change_type", "resolution_approved"),
        ("change_application", "shareholder_resolution", "new_legal_representative_id"),
        ("company_abnormal", "document_forgery"),
        "https://gswsdj.zjzwfw.gov.cn/entrance_unite.html?siteCode=330105000000",
        "浙江企业在线",
        "浙江省市场监督管理局",
        {"city": "杭州", "company_status": "normal", "change_type": "legal_representative", "resolution_approved": True},
    ),
    MatterTemplate(
        "construction_permit",
        "工程建设",
        "建设工程施工许可证",
        "项目准备开工，想办理建设工程施工许可证。",
        ("city", "land_approval", "planning_approval", "contractor_confirmed"),
        ("permit_application", "land_certificate", "construction_contract"),
        ("unapproved_land", "contractor_blacklist"),
        "https://www.hzsc.gov.cn/art/2020/7/29/art_1229631671_4101150.html",
        "建设工程施工许可证申请服务指南",
        "杭州市上城区住房和城市建设局",
        {"city": "杭州", "land_approval": True, "planning_approval": True, "contractor_confirmed": True},
    ),
    MatterTemplate(
        "lawyer_practice_registration",
        "公共法律服务",
        "首次申请律师执业",
        "我通过了法律职业资格考试，想首次申请律师执业。",
        ("city", "legal_qualification", "internship_months", "conduct_status"),
        ("application_form", "qualification_certificate", "internship_certificate", "employment_contract"),
        ("criminal_record", "dismissed_from_public_office"),
        "https://sfj.ezhou.gov.cn/ztzl_934/ggflfw/202506/t20250613_709104.html",
        "公共法律服务办事指南（律师服务）",
        "鄂州市司法局",
        {"city": "鄂州", "legal_qualification": True, "internship_months": 12, "conduct_status": "good"},
        ({"slot": "internship_months", "operator": ">=", "value": 12, "failure_reason": "实习期限不足一年"},),
    ),
)


def source_hash(template: MatterTemplate) -> str:
    return _source_hash(template.source_title, template.source_url)

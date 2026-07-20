from gov_agent_rl.data_builder import build_cases
from scripts.audit_case_consistency import audit_records


def test_diverse_v2_labels_are_supported_by_tool_results():
    report = audit_records(
        [case.model_dump(mode="json") for case in build_cases(diverse=True)]
    )
    assert report["count"] == 1200
    assert report["conflict_count"] == 0


def test_legacy_audit_exposes_known_ineligible_label_conflicts():
    report = audit_records(
        [case.model_dump(mode="json") for case in build_cases(diverse=False)]
    )
    assert report["conflict_count"] == 20
    assert report["conflicts_by_matter"] == {"provident_fund_loan": 20}
    assert report["conflicts_by_scenario"] == {"ineligible": 20}

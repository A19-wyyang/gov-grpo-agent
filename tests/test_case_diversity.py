from gov_agent_rl.data_builder import build_cases
from scripts.audit_case_diversity import audit_cases


def test_every_matter_scenario_group_has_unique_meaningful_cases():
    records = [case.model_dump(mode="json") for case in build_cases()]
    report = audit_cases(records)
    assert report["count"] == 1200
    assert report["full_unique"] == 1200
    assert report["minimum_group_full_unique_rate"] == 1.0
    assert report["minimum_group_visible_unique_rate"] == 1.0


def test_case_generation_is_seed_deterministic_but_seed_changes_order():
    first = [case.model_dump_json() for case in build_cases(seed=42)]
    repeated = [case.model_dump_json() for case in build_cases(seed=42)]
    different = [case.model_dump_json() for case in build_cases(seed=43)]
    assert first == repeated
    assert first != different


def test_legacy_variant_remains_rebuildable_for_historical_rescoring():
    legacy = [case.model_dump(mode="json") for case in build_cases(diverse=False)]
    report = audit_cases(legacy)
    assert report["count"] == 1200
    assert report["visible_unique"] == 36
    assert report["full_unique"] < 1200

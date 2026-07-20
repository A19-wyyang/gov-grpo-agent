import json

import pytest

from scripts.build_scenario_curriculum import (
    build_curriculum,
    scenario_scores,
    write_curriculum,
)


def _row(case_id: str, scenario: str, matter: str = "matter"):
    return {
        "prompt": [{"role": "user", "content": case_id}],
        "extra_info": {
            "case_id": case_id,
            "scenario_type": scenario,
            "matter_id": matter,
            "split": "train",
        },
    }


def test_curriculum_oversamples_low_process_success_with_a_global_cap():
    rows = [
        *[_row(f"missing-{index}", "missing_information") for index in range(4)],
        *[_row(f"success-{index}", "success") for index in range(4)],
    ]
    metrics = {
        "scenario_metrics": {
            "missing_information": {"process_pass_at_k": 0.0},
            "success": {"process_pass_at_k": 1.0},
        }
    }
    output, manifest = build_curriculum(
        rows,
        metrics,
        max_expansion=1.5,
        seed=7,
    )
    assert len(output) == 12
    assert manifest["extra_rows_by_scenario"] == {
        "missing_information": 4,
        "success": 0,
    }
    assert manifest["output_scenario_counts"]["missing_information"] == 8
    assert all("curriculum_repeat_index" in row for row in output)
    assert max(row["curriculum_repeat_index"] for row in output) == 1


def test_curriculum_uses_metric_priority_and_is_deterministic():
    metrics = {
        "scenario_metrics": {
            "risk": {
                "process_pass_at_1": 0.10,
                "process_pass_at_k": 0.25,
                "pass_at_1": 1.0,
            }
        }
    }
    scores, sources = scenario_scores(metrics)
    assert scores == {"risk": 0.10}
    assert sources == {"risk": "process_pass_at_1"}

    rows = [_row(f"risk-{index}", "risk") for index in range(4)]
    first, _ = build_curriculum(rows, metrics, seed=11)
    second, _ = build_curriculum(rows, metrics, seed=11)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_curriculum_rejects_test_leakage_and_missing_scenarios():
    test_row = _row("case", "risk")
    test_row["extra_info"]["split"] = "test"
    metrics = {"scenario_metrics": {"risk": {"pass_at_1": 0.0}}}
    with pytest.raises(ValueError, match="train rows only"):
        build_curriculum([test_row], metrics)

    with pytest.raises(ValueError, match="supported scenario success"):
        build_curriculum([_row("case", "risk")], {"scenario_metrics": {}})


def test_curriculum_writes_jsonl_parquet_and_hashed_manifest(tmp_path):
    source = tmp_path / "source.jsonl"
    metrics_path = tmp_path / "validation.json"
    source.write_text("{}\n", encoding="utf-8")
    metrics_path.write_text("{}\n", encoding="utf-8")
    rows = [_row("case", "risk")]
    manifest = {"dataset_variant": "scenario_curriculum_v3"}
    output_dir = tmp_path / "curriculum"
    write_curriculum(rows, manifest, output_dir, source, metrics_path)

    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "train.parquet").is_file()
    payload = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(payload["source_train_sha256"]) == 64
    assert len(payload["validation_metrics_sha256"]) == 64
    assert len(payload["train_parquet_sha256"]) == 64

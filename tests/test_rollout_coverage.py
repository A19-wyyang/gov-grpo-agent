import json

from scripts.check_rollout_coverage import check_coverage


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_rollout_coverage_requires_every_case_and_exact_k(tmp_path):
    cases = tmp_path / "cases.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    _write_jsonl(cases, [{"case_id": "a"}, {"case_id": "b"}])
    _write_jsonl(
        rollouts,
        [
            {"case_id": case_id, "step": 25, "sample": sample}
            for case_id in ("a", "b")
            for sample in range(4)
        ],
    )
    complete = check_coverage(rollouts, cases, 4, expected_step=25)
    assert complete["ok"]
    assert complete["rollouts"] == 8

    _write_jsonl(
        rollouts,
        [
            {"case_id": "a", "step": 25, "sample": sample}
            for sample in range(4)
        ]
        + [{"case_id": "b", "step": 25, "sample": 0}],
    )
    partial = check_coverage(rollouts, cases, 4, expected_step=25)
    assert not partial["ok"]
    assert partial["wrong_count_cases"] == 1

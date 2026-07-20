#!/usr/bin/env python3
"""Validate exact case and rollout coverage before reporting evaluation metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            records.append(record)
    return records


def check_coverage(
    rollout_path: Path,
    cases_path: Path,
    expected_rollouts_per_case: int,
    expected_step: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        rollout_records = _read_jsonl(rollout_path)
    except (OSError, ValueError) as exc:
        rollout_records = []
        errors.append(str(exc))
    try:
        case_records = _read_jsonl(cases_path)
    except (OSError, ValueError) as exc:
        case_records = []
        errors.append(str(exc))

    expected_case_ids = {
        str(record.get("case_id", "")) for record in case_records
        if record.get("case_id")
    }
    counts = Counter(
        str(record.get("case_id", "")) for record in rollout_records
    )
    actual_case_ids = {case_id for case_id in counts if case_id}
    missing = sorted(expected_case_ids - actual_case_ids)
    unknown = sorted(actual_case_ids - expected_case_ids)
    empty_case_ids = counts.get("", 0)
    wrong_counts = {
        case_id: counts.get(case_id, 0)
        for case_id in sorted(expected_case_ids)
        if counts.get(case_id, 0) != expected_rollouts_per_case
    }
    if not expected_case_ids:
        errors.append("case specification contains no non-empty case_id")
    if missing:
        errors.append(f"missing cases: {missing[:10]}")
    if unknown:
        errors.append(f"unknown cases: {unknown[:10]}")
    if empty_case_ids:
        errors.append(f"rollouts with empty case_id: {empty_case_ids}")
    if wrong_counts:
        errors.append(
            "wrong rollout counts: "
            + str(list(wrong_counts.items())[:10])
        )
    if expected_step is not None:
        bad_steps = Counter(
            str(record.get("step"))
            for record in rollout_records
            if record.get("step") is not None
            and str(record.get("step")) != str(expected_step)
        )
        if bad_steps:
            errors.append(f"unexpected steps: {dict(bad_steps)}")

    expected_total = len(expected_case_ids) * expected_rollouts_per_case
    if len(rollout_records) != expected_total:
        errors.append(
            f"rollout record count {len(rollout_records)} != expected {expected_total}"
        )
    return {
        "ok": not errors,
        "cases": len(expected_case_ids),
        "rollouts": len(rollout_records),
        "expected_rollouts_per_case": expected_rollouts_per_case,
        "missing_cases": len(missing),
        "unknown_cases": len(unknown),
        "wrong_count_cases": len(wrong_counts),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--rollouts-per-case", type=int, required=True)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if args.rollouts_per_case <= 0:
        raise ValueError("--rollouts-per-case must be positive")
    result = check_coverage(
        args.rollouts,
        args.cases,
        args.rollouts_per_case,
        args.step,
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

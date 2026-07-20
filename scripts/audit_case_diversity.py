#!/usr/bin/env python3
"""Report effective prompt/truth diversity for generated government cases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fingerprint(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    return json.dumps(
        {field: record[field] for field in fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def audit_cases(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no cases")
    visible_fields = ("matter_id", "user_request", "visible_slots")
    full_fields = (
        "matter_id",
        "scenario_type",
        "user_request",
        "visible_slots",
        "hidden_truth",
        "expected_result",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["matter_id"]), str(record["scenario_type"]))].append(
            record
        )
    group_rows = []
    for (matter, scenario), items in sorted(groups.items()):
        visible_unique = len({_fingerprint(item, visible_fields) for item in items})
        full_unique = len({_fingerprint(item, full_fields) for item in items})
        group_rows.append(
            {
                "matter_id": matter,
                "scenario_type": scenario,
                "count": len(items),
                "visible_unique": visible_unique,
                "visible_unique_rate": visible_unique / len(items),
                "full_unique": full_unique,
                "full_unique_rate": full_unique / len(items),
            }
        )
    return {
        "count": len(records),
        "visible_unique": len(
            {_fingerprint(record, visible_fields) for record in records}
        ),
        "full_unique": len(
            {_fingerprint(record, full_fields) for record in records}
        ),
        "minimum_group_visible_unique_rate": min(
            row["visible_unique_rate"] for row in group_rows
        ),
        "minimum_group_full_unique_rate": min(
            row["full_unique_rate"] for row in group_rows
        ),
        "groups": group_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-full-unique", action="store_true")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = audit_cases(records)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_full_unique and report["minimum_group_full_unique_rate"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import GovCase


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")


def load_cases(cases_dir: Path) -> list[GovCase]:
    cases = []
    for path in sorted(cases_dir.glob("*.json")):
        data = read_json(path)
        cases.append(GovCase(**data))
    return cases

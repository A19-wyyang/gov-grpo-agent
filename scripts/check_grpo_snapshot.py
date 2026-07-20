#!/usr/bin/env python3
"""Verify that a veRL validation/checkpoint snapshot is durable and complete."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def check_snapshot(
    validation_path: Path,
    checkpoint_dir: Path,
    step: int,
    expected_cases: int,
    world_size: int,
    min_age_seconds: float = 0,
) -> dict[str, object]:
    errors: list[str] = []
    records: list[dict[str, object]] = []
    if not validation_path.is_file():
        errors.append(f"validation file missing: {validation_path}")
    else:
        with validation_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"validation line {line_number} is invalid JSON: {exc}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"validation line {line_number} is not an object")
                    continue
                records.append(record)

    case_ids = [str(record.get("case_id", "")) for record in records]
    if len(records) != expected_cases:
        errors.append(
            f"validation record count {len(records)} != expected {expected_cases}"
        )
    if len(set(case_ids)) != expected_cases or any(not case_id for case_id in case_ids):
        errors.append(
            f"validation unique non-empty case count {len({item for item in case_ids if item})} "
            f"!= expected {expected_cases}"
        )
    steps: set[int] = set()
    for index, record in enumerate(records, start=1):
        try:
            steps.add(int(record.get("step", -1)))
        except (TypeError, ValueError):
            errors.append(
                f"validation record {index} has invalid step: {record.get('step')!r}"
            )
    if records and steps != {step}:
        errors.append(f"validation steps {sorted(steps)} != expected [{step}]")

    required_files = [checkpoint_dir / "data.pt"]
    for rank in range(world_size):
        required_files.extend(
            [
                checkpoint_dir / "actor" / f"model_world_size_{world_size}_rank_{rank}.pt",
                checkpoint_dir / "actor" / f"optim_world_size_{world_size}_rank_{rank}.pt",
                checkpoint_dir
                / "actor"
                / f"extra_state_world_size_{world_size}_rank_{rank}.pt",
            ]
        )
    required_files.extend(
        [
            checkpoint_dir / "actor" / "fsdp_config.json",
            checkpoint_dir / "actor" / "lora_train_meta.json",
            checkpoint_dir / "actor" / "huggingface" / "config.json",
            checkpoint_dir / "actor" / "huggingface" / "tokenizer.json",
        ]
    )
    now = time.time()
    durable_files = [validation_path, *required_files]
    for path in required_files:
        try:
            is_nonempty = path.is_file() and path.stat().st_size > 0
        except OSError:
            is_nonempty = False
        if not is_nonempty:
            errors.append(f"checkpoint file missing or empty: {path}")
    if min_age_seconds > 0:
        for path in durable_files:
            try:
                age_seconds = now - path.stat().st_mtime
            except OSError:
                continue
            if age_seconds < min_age_seconds:
                errors.append(
                    f"snapshot file is too fresh ({age_seconds:.1f}s < "
                    f"{min_age_seconds:.1f}s): {path}"
                )

    model_files = list((checkpoint_dir / "actor").glob("model_world_size_*_rank_*.pt"))
    if len(model_files) != world_size:
        errors.append(
            f"model shard count {len(model_files)} != expected world size {world_size}"
        )

    result: dict[str, object] = {
        "ok": not errors,
        "step": step,
        "validation_records": len(records),
        "unique_cases": len({case_id for case_id in case_ids if case_id}),
        "checkpoint_dir": str(checkpoint_dir),
        "model_shards": len(model_files),
        "min_age_seconds": min_age_seconds,
        "errors": errors,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-cases", type=int, default=200)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--min-age-seconds", type=float, default=30)
    args = parser.parse_args()
    result = check_snapshot(
        args.validation,
        args.checkpoint,
        args.step,
        args.expected_cases,
        args.world_size,
        args.min_age_seconds,
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

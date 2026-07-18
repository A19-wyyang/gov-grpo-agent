from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_builder import write_dataset
from .evaluation import evaluate_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gov_agent_rl",
        description="Government-service SFT, veRL GRPO data and evaluation tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser(
        "build-data", help="Build the versioned 1,200-case SFT/GRPO dataset."
    )
    data_parser.add_argument("--out", type=Path, default=Path("data/processed"))
    data_parser.add_argument("--seed", type=int, default=42)
    data_parser.add_argument(
        "--no-parquet", action="store_true", help="Write JSONL only."
    )

    validate_parser = subparsers.add_parser(
        "validate-data", help="Validate generated cases and split isolation."
    )
    validate_parser.add_argument("--data", type=Path, default=Path("data/processed"))

    eval_parser = subparsers.add_parser(
        "evaluate", help="Aggregate rollout evaluation JSONL."
    )
    eval_parser.add_argument("--input", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-data":
        counts = write_dataset(
            args.out, seed=args.seed, write_parquet=not args.no_parquet
        )
        print(json.dumps({"output": str(args.out), "splits": counts}, ensure_ascii=False))
    elif args.command == "validate-data":
        result = validate_generated_data(args.data)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "evaluate":
        result = evaluate_jsonl(args.input, args.out)
        print(json.dumps(result, ensure_ascii=False))
def validate_generated_data(data_dir: Path) -> dict[str, object]:
    from .schema import CaseSpec

    template_splits: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    case_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.cases.jsonl"
        rows = [
            CaseSpec.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        counts[split] = len(rows)
        for case in rows:
            if case.case_id in case_ids:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            case_ids.add(case.case_id)
            template_splits.setdefault(case.matter_id, set()).add(split)
    leaked = {
        template: sorted(splits)
        for template, splits in template_splits.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"matter split leakage: {leaked}")
    if sum(counts.values()) != 1200:
        raise ValueError(f"expected 1200 cases, got {counts}")
    return {"valid": True, "counts": counts, "matter_count": len(template_splits)}

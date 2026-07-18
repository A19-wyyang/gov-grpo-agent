from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_builder import write_dataset
from .evaluation import evaluate_jsonl
from .experiments import run_training_comparison
from .grpo import export_grpo_file
from .report import generate_report
from .rollout import rollout_to_file
from .scoring import score_file


DEFAULT_CASES_DIR = Path("data/cases")
DEFAULT_RUN_DIR = Path("runs/demo")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gov_agent_rl",
        description="Local government-service Agentic RL data-flow demo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the full local demo flow.")
    demo_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    demo_parser.add_argument("--out", type=Path, default=DEFAULT_RUN_DIR)

    rollout_parser = subparsers.add_parser("rollout", help="Generate trajectories.")
    rollout_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    rollout_parser.add_argument("--out", type=Path, default=DEFAULT_RUN_DIR)

    verify_parser = subparsers.add_parser("verify", help="Score trajectories.")
    verify_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    verify_parser.add_argument("--trajectories", type=Path, required=True)
    verify_parser.add_argument("--out", type=Path, required=True)
    verify_parser.add_argument(
        "--profile",
        choices=["hardened", "collapse_prone"],
        default="hardened",
    )

    grpo_parser = subparsers.add_parser("export-grpo", help="Export GRPO groups.")
    grpo_parser.add_argument("--scores", type=Path, required=True)
    grpo_parser.add_argument("--out", type=Path, required=True)

    report_parser = subparsers.add_parser("report", help="Generate Markdown report.")
    report_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    report_parser.add_argument("--trajectories", type=Path, required=True)
    report_parser.add_argument("--scores", type=Path, required=True)
    report_parser.add_argument("--groups", type=Path, required=True)
    report_parser.add_argument("--out", type=Path, required=True)

    train_parser = subparsers.add_parser(
        "train-sim",
        help="Run the policy-collapse and reward-hacking training simulation.",
    )
    train_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_DIR)
    train_parser.add_argument("--out", type=Path, default=DEFAULT_RUN_DIR)

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
    if args.command == "demo":
        run_demo(args.cases, args.out)
    elif args.command == "rollout":
        output = rollout_to_file(args.cases, args.out)
        print(f"wrote {output}")
    elif args.command == "verify":
        output = score_file(args.cases, args.trajectories, args.out, args.profile)
        print(f"wrote {output}")
    elif args.command == "export-grpo":
        output = export_grpo_file(args.scores, args.out)
        print(f"wrote {output}")
    elif args.command == "report":
        output = generate_report(
            args.cases,
            args.trajectories,
            args.scores,
            args.groups,
            args.out,
        )
        print(f"wrote {output}")
    elif args.command == "train-sim":
        output = run_training_comparison(args.cases, args.out)
        print(f"wrote {output}")
    elif args.command == "build-data":
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


def run_demo(cases_dir: Path, out_dir: Path) -> None:
    trajectories_path = rollout_to_file(cases_dir, out_dir)
    scores_path = score_file(
        cases_dir,
        trajectories_path,
        out_dir / "scored.jsonl",
    )
    groups_path = export_grpo_file(scores_path, out_dir / "grpo_groups.jsonl")
    report_path = generate_report(
        cases_dir,
        trajectories_path,
        scores_path,
        groups_path,
        out_dir / "report.md",
    )
    experiment_report_path = run_training_comparison(cases_dir, out_dir)
    print(f"wrote {trajectories_path}")
    print(f"wrote {scores_path}")
    print(f"wrote {groups_path}")
    print(f"wrote {report_path}")
    print(f"wrote {experiment_report_path}")


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

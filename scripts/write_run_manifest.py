#!/usr/bin/env python3
"""Create an immutable, secret-free manifest before a GRPO run starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REWARD_DEFAULTS = {
    "GOV_MISSING_TOOL_PENALTY": "0.30",
    "GOV_MISSING_TOOL_HARD_GATE": "0",
    "GOV_DECISION_GATE_CEILING": "0.20",
    "GOV_PROCESS_GATE_CEILING": "0.20",
    "GOV_INVALID_SLOT_PENALTY": "0.08",
    "GOV_ILLEGAL_ACTION_PENALTY": "0.15",
    "GOV_ACTION_AFTER_DONE_PENALTY": "0.10",
    "GOV_REPEATED_TOOL_PENALTY": "0.05",
    "GOV_TOOL_ORDER_PENALTY": "0.10",
    "GOV_EARLY_ELIGIBILITY_PENALTY": "0.15",
    "GOV_HARD_FACT_WEIGHT": "0.65",
    "GOV_PROCESS_WEIGHT": "0.25",
    "GOV_EXPRESSION_WEIGHT": "0.10",
    "GOV_JUDGE_FAILURE_SCORE": "0.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory has no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(project_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    *,
    project_dir: Path,
    experiment: str,
    train_file: Path,
    val_file: Path,
    model: str,
    sft_adapter: Path,
    tool_config: Path,
    training: dict[str, Any],
) -> dict[str, Any]:
    for path in (train_file, val_file, tool_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not sft_adapter.is_dir():
        raise FileNotFoundError(sft_adapter)
    dataset_manifest_path = train_file.parent / "manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    dataset_manifest = json.loads(
        dataset_manifest_path.read_text(encoding="utf-8")
    )
    return {
        "schema_version": 1,
        "experiment": experiment,
        "code": {"git_commit": _git_commit(project_dir)},
        "dataset": {
            "manifest": dataset_manifest,
            "train_file": str(train_file.resolve()),
            "train_sha256": sha256_file(train_file),
            "validation_file": str(val_file.resolve()),
            "validation_sha256": sha256_file(val_file),
        },
        "model": {
            "base": model,
            "sft_adapter": str(sft_adapter.resolve()),
            "sft_adapter_sha256": sha256_directory(sft_adapter),
        },
        "tool": {
            "config": str(tool_config.resolve()),
            "config_sha256": sha256_file(tool_config),
        },
        "reward": {
            name: os.getenv(name, default)
            for name, default in REWARD_DEFAULTS.items()
        },
        "judge": {
            "base_url": os.getenv("GOV_JUDGE_BASE_URL"),
            "model": os.getenv("GOV_JUDGE_MODEL"),
            "required": os.getenv("GOV_JUDGE_REQUIRED", "0"),
            "enable_thinking": os.getenv("GOV_JUDGE_ENABLE_THINKING", "0"),
        },
        "training": training,
    }


def write_or_verify_manifest(path: Path, manifest: dict[str, Any]) -> str:
    rendered = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(rendered, encoding="utf-8")
        return "created"
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == manifest:
        return "verified"
    candidate_path = path.with_name(path.stem + ".candidate.json")
    candidate_path.write_text(rendered, encoding="utf-8")
    raise ValueError(
        f"run manifest mismatch for existing experiment: {path}; "
        f"candidate written to {candidate_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sft-adapter", type=Path, required=True)
    parser.add_argument("--tool-config", type=Path, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--actor-lr", type=float, required=True)
    parser.add_argument("--clip-ratio", type=float, required=True)
    parser.add_argument("--clip-ratio-low", type=float, required=True)
    parser.add_argument("--clip-ratio-high", type=float, required=True)
    parser.add_argument("--kl-loss-coef", type=float, required=True)
    parser.add_argument("--entropy-coeff", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rollout-seed", type=int, required=True)
    parser.add_argument("--eval-rollout-n", type=int, required=True)
    parser.add_argument("--eval-temperature", type=float, required=True)
    parser.add_argument("--eval-top-p", type=float, required=True)
    parser.add_argument("--eval-do-sample", required=True)
    parser.add_argument(
        "--loss-agg-mode",
        choices=(
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        ),
        required=True,
    )
    parser.add_argument("--gen-batch-size", type=int, required=True)
    parser.add_argument("--filter-groups-enable", required=True)
    parser.add_argument("--filter-groups-metric", required=True)
    parser.add_argument("--filter-max-gen-batches", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--total-epochs", type=float, required=True)
    parser.add_argument("--trainer-overrides", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        project_dir=args.project_dir,
        experiment=args.experiment,
        train_file=args.train_file,
        val_file=args.val_file,
        model=args.model,
        sft_adapter=args.sft_adapter,
        tool_config=args.tool_config,
        training={
            "rollout_n": args.rollout_n,
            "train_batch_size": args.train_batch_size,
            "ppo_mini_batch_size": args.ppo_mini_batch_size,
            "actor_lr": args.actor_lr,
            "clip_ratio": args.clip_ratio,
            "clip_ratio_low": args.clip_ratio_low,
            "clip_ratio_high": args.clip_ratio_high,
            "kl_loss_coef": args.kl_loss_coef,
            "entropy_coeff": args.entropy_coeff,
            "seed": args.seed,
            "rollout_seed": args.rollout_seed,
            "eval_rollout_n": args.eval_rollout_n,
            "eval_temperature": args.eval_temperature,
            "eval_top_p": args.eval_top_p,
            "eval_do_sample": args.eval_do_sample.lower()
            in {"1", "true", "yes"},
            "loss_agg_mode": args.loss_agg_mode,
            "gen_batch_size": args.gen_batch_size,
            "filter_groups_enable": (
                args.filter_groups_enable.lower()
                in {"1", "true", "yes"}
            ),
            "filter_groups_metric": args.filter_groups_metric,
            "filter_max_gen_batches": args.filter_max_gen_batches,
            "max_model_len": args.max_model_len,
            "total_epochs": args.total_epochs,
            "trainer_overrides": args.trainer_overrides,
        },
    )
    status = write_or_verify_manifest(args.output, manifest)
    print(json.dumps({"status": status, "manifest": str(args.output)}))


if __name__ == "__main__":
    main()

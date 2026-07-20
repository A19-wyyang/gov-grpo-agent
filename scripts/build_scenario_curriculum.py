#!/usr/bin/env python3
"""Build a validation-driven, scenario-balanced GRPO training curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from gov_agent_rl.evaluation import evaluate_rows


METRIC_PRIORITY = (
    "process_pass_at_1",
    "process_pass_at_k",
    "process_success_at_k",
    "safe_pass_at_1",
    "safe_pass_at_k",
    "safe_success_at_k",
    "pass_at_k",
    "pass_at_1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_validation_metrics(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return evaluate_rows(read_jsonl(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "scenario_metrics" not in payload:
        raise ValueError(f"metrics file has no scenario_metrics: {path}")
    return payload


def scenario_scores(metrics: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    for scenario, values in metrics.get("scenario_metrics", {}).items():
        if not isinstance(values, dict):
            continue
        for metric in METRIC_PRIORITY:
            value = values.get(metric)
            if value is not None:
                scores[str(scenario)] = max(0.0, min(1.0, float(value)))
                sources[str(scenario)] = metric
                break
    if not scores:
        raise ValueError("no supported scenario success metric was found")
    return scores, sources


def _allocate_extras(
    base_counts: Counter[str],
    multipliers: dict[str, float],
    budget: int,
) -> dict[str, int]:
    raw = {
        scenario: count * max(0.0, multipliers.get(scenario, 1.0) - 1.0)
        for scenario, count in base_counts.items()
    }
    raw_total = sum(raw.values())
    if raw_total <= 0 or budget <= 0:
        return {scenario: 0 for scenario in base_counts}
    scale = min(1.0, budget / raw_total)
    scaled = {scenario: value * scale for scenario, value in raw.items()}
    allocated = {scenario: math.floor(value) for scenario, value in scaled.items()}
    remaining = min(budget, round(sum(scaled.values()))) - sum(allocated.values())
    ranking = sorted(
        scaled,
        key=lambda scenario: (
            scaled[scenario] - allocated[scenario],
            raw[scenario],
            scenario,
        ),
        reverse=True,
    )
    for scenario in ranking[:remaining]:
        allocated[scenario] += 1
    return allocated


def build_curriculum(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    alpha: float = 1.0,
    max_multiplier: float = 2.0,
    max_expansion: float = 1.5,
    target_success: float = 1.0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("training dataset is empty")
    if alpha < 0 or max_multiplier < 1 or max_expansion < 1:
        raise ValueError("invalid curriculum scaling parameters")

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matters_by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        extra = row.get("extra_info", {})
        split = str(extra.get("split", ""))
        if split != "train":
            raise ValueError(
                f"curriculum source must contain train rows only, found split={split!r}"
            )
        scenario = str(extra.get("scenario_type", "unknown"))
        matter = str(extra.get("matter_id", "unknown"))
        by_scenario[scenario].append(row)
        matters_by_scenario[scenario][matter] += 1

    scores, metric_sources = scenario_scores(metrics)
    missing_scores = sorted(set(by_scenario) - set(scores))
    if missing_scores:
        raise ValueError(
            f"validation metrics missing training scenarios: {missing_scores}"
        )
    multipliers = {
        scenario: min(
            max_multiplier,
            1.0 + alpha * max(0.0, target_success - scores[scenario]),
        )
        for scenario in by_scenario
    }
    base_counts = Counter(
        {
            scenario: len(scenario_rows)
            for scenario, scenario_rows in by_scenario.items()
        }
    )
    budget = math.floor(len(rows) * (max_expansion - 1.0))
    extras = _allocate_extras(base_counts, multipliers, budget)

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    for row in rows:
        item = deepcopy(row)
        item["curriculum_repeat_index"] = 0
        output.append(item)

    for scenario in sorted(by_scenario):
        pool = sorted(
            by_scenario[scenario],
            key=lambda row: str(row["extra_info"].get("case_id", "")),
        )
        rng.shuffle(pool)
        for repeat_index in range(extras[scenario]):
            item = deepcopy(pool[repeat_index % len(pool)])
            item["curriculum_repeat_index"] = (
                repeat_index // len(pool)
            ) + 1
            output.append(item)
    rng.shuffle(output)

    output_counts = Counter(
        str(row["extra_info"].get("scenario_type", "unknown"))
        for row in output
    )
    manifest = {
        "dataset_variant": "scenario_curriculum_v3",
        "seed": seed,
        "base_count": len(rows),
        "output_count": len(output),
        "max_expansion": max_expansion,
        "target_success": target_success,
        "alpha": alpha,
        "max_multiplier": max_multiplier,
        "selection_split": "validation",
        "base_scenario_counts": dict(sorted(base_counts.items())),
        "output_scenario_counts": dict(sorted(output_counts.items())),
        "validation_scenario_scores": dict(sorted(scores.items())),
        "validation_metric_by_scenario": dict(sorted(metric_sources.items())),
        "scenario_multipliers": {
            key: round(value, 6) for key, value in sorted(multipliers.items())
        },
        "extra_rows_by_scenario": dict(sorted(extras.items())),
        "base_matter_counts_by_scenario": {
            scenario: dict(sorted(counts.items()))
            for scenario, counts in sorted(matters_by_scenario.items())
        },
    }
    return output, manifest


def write_curriculum(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
    source_path: Path,
    metrics_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write the GRPO parquet") from exc
    parquet_path = output_dir / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    payload = {
        **manifest,
        "source_train_file": str(source_path.resolve()),
        "source_train_sha256": sha256_file(source_path),
        "validation_metrics_file": str(metrics_path.resolve()),
        "validation_metrics_sha256": sha256_file(metrics_path),
        "train_jsonl_sha256": sha256_file(jsonl_path),
        "train_parquet_sha256": sha256_file(parquet_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-metrics", type=Path, required=True)
    parser.add_argument("--metrics-split", choices=("validation",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-multiplier", type=float, default=2.0)
    parser.add_argument("--max-expansion", type=float, default=1.5)
    parser.add_argument("--target-success", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    curriculum, manifest = build_curriculum(
        read_jsonl(args.train_jsonl),
        load_validation_metrics(args.validation_metrics),
        alpha=args.alpha,
        max_multiplier=args.max_multiplier,
        max_expansion=args.max_expansion,
        target_success=args.target_success,
        seed=args.seed,
    )
    write_curriculum(
        curriculum,
        manifest,
        args.output_dir,
        args.train_jsonl,
        args.validation_metrics,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

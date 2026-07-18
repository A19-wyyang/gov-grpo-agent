from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl


def build_grpo_groups(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_case[row["case_id"]].append(row)

    groups = []
    for case_id, rows in sorted(by_case.items()):
        mean_reward = sum(row["reward"] for row in rows) / len(rows)
        reward_variance = sum((row["reward"] - mean_reward) ** 2 for row in rows) / len(rows)
        reward_std = sqrt(reward_variance)
        ranked = sorted(rows, key=lambda row: row["reward"], reverse=True)
        group_rows = []
        for rank, row in enumerate(ranked, start=1):
            group_rows.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "policy_name": row["policy_name"],
                    "reward": row["reward"],
                    "rank": rank,
                    "raw_advantage": round(row["reward"] - mean_reward, 4),
                    "advantage": round(
                        (row["reward"] - mean_reward) / reward_std if reward_std else 0.0,
                        4,
                    ),
                    "verdict": row["verdict"],
                    "penalties": row["penalties"],
                }
            )
        groups.append(
            {
                "case_id": case_id,
                "mean_reward": round(mean_reward, 4),
                "reward_std": round(reward_std, 4),
                "trajectories": group_rows,
            }
        )
    return groups


def export_grpo_file(scores_path: Path, output_path: Path) -> Path:
    groups = build_grpo_groups(read_jsonl(scores_path))
    write_jsonl(output_path, groups)
    return output_path

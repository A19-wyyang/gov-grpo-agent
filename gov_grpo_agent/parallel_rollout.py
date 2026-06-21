import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseShard:
    gpu_id: int
    case_offset: int
    case_count: int


@dataclass(frozen=True)
class WorkerCommand:
    gpu_id: int
    output_dir: Path
    log_path: Path
    env: dict
    args: list


def plan_case_shards(gpu_ids, total_cases):
    gpu_list = [int(gpu_id) for gpu_id in gpu_ids]
    base = total_cases // len(gpu_list)
    remainder = total_cases % len(gpu_list)
    shards = []
    offset = 0
    for index, gpu_id in enumerate(gpu_list):
        count = base + (1 if index < remainder else 0)
        shards.append(CaseShard(gpu_id=gpu_id, case_offset=offset, case_count=count))
        offset += count
    return shards


def build_worker_commands(
    gpu_ids,
    total_cases,
    model_name_or_path,
    adapter_path,
    output_root,
    rollout_group_size,
    max_turns,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    python_executable=None,
):
    executable = python_executable or sys.executable
    root = Path(output_root)
    commands = []
    for shard in plan_case_shards(gpu_ids, total_cases):
        output_dir = root / f"gpu{shard.gpu_id}"
        log_path = output_dir / "rollout.log"
        env = {
            "CUDA_VISIBLE_DEVICES": str(shard.gpu_id),
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
        }
        args = [
            executable,
            "-m",
            "gov_grpo_agent.model_rollout",
            "--model-name-or-path",
            model_name_or_path,
            "--adapter-path",
            adapter_path,
            "--output-dir",
            output_dir.as_posix(),
            "--case-offset",
            str(shard.case_offset),
            "--case-count",
            str(shard.case_count),
            "--rollout-group-size",
            str(rollout_group_size),
            "--max-turns",
            str(max_turns),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
        ]
        if do_sample:
            args.append("--do-sample")
        else:
            args.append("--no-sample")
        commands.append(
            WorkerCommand(
                gpu_id=shard.gpu_id,
                output_dir=output_dir,
                log_path=log_path,
                env=env,
                args=args,
            )
        )
    return commands


def launch_workers(commands, dry_run=False):
    launched = []
    for command in commands:
        command.output_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            launched.append({"gpu_id": command.gpu_id, "pid": None, "log_path": str(command.log_path)})
            continue
        env = os.environ.copy()
        env.update(command.env)
        log_handle = command.log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command.args,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.close()
        launched.append({"gpu_id": command.gpu_id, "pid": process.pid, "log_path": str(command.log_path)})
    return launched


def _parse_gpu_ids(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch multi-GPU model rollout workers.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--total-cases", type=int, default=200)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", default="artifacts/qwen3_8b_sft_lora")
    parser.add_argument("--output-root", default="artifacts/parallel_model_rollout")
    parser.add_argument("--rollout-group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--no-sample", action="store_false", dest="do_sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    commands = build_worker_commands(
        gpu_ids=_parse_gpu_ids(args.gpus),
        total_cases=args.total_cases,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        output_root=Path(args.output_root),
        rollout_group_size=args.rollout_group_size,
        max_turns=args.max_turns,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    launched = launch_workers(commands, dry_run=args.dry_run)
    print(json.dumps(launched, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print(f"Logs: tail -f {Path(args.output_root).as_posix()}/gpu*/rollout.log")


if __name__ == "__main__":
    main()

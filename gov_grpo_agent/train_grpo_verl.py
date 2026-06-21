import argparse
import json
from pathlib import Path

from gov_grpo_agent.verl_config import build_verl_grpo_command, write_verl_grpo_config
from gov_grpo_agent.verl_data import convert_grpo_jsonl_to_verl_parquet


def prepare_verl_training_job(
    input_jsonl,
    work_dir,
    model_path="Qwen/Qwen3-8B",
    n_rollout=4,
    total_epochs=1,
):
    root = Path(work_dir)
    data_dir = root / "data"
    config_dir = root / "configs"
    save_dir = root / "checkpoints"
    tensorboard_dir = root / "tensorboard"
    train_parquet = data_dir / "train.parquet"
    data_report_path = data_dir / "data_report.json"
    config_path = config_dir / "verl_grpo_qwen3_8b.yaml"
    run_script = root / "run_verl_grpo.sh"
    manifest_path = root / "manifest.json"

    data_report = convert_grpo_jsonl_to_verl_parquet(
        input_path=input_jsonl,
        output_path=train_parquet,
        report_path=data_report_path,
    )
    write_verl_grpo_config(
        output_path=config_path,
        model_path=model_path,
        train_files=str(train_parquet),
        val_files=str(train_parquet),
        save_dir=str(save_dir),
        reward_path="gov_grpo_agent/verl_reward.py",
        reward_name="compute_score",
        n_rollout=n_rollout,
        total_epochs=total_epochs,
    )
    command = build_verl_grpo_command(config_path)
    _write_run_script(run_script, command)
    manifest = {
        "input_jsonl": str(input_jsonl),
        "train_parquet": str(train_parquet),
        "data_report": data_report,
        "config": str(config_path),
        "run_script": str(run_script),
        "tensorboard_dir": str(tensorboard_dir),
        "checkpoint_dir": str(save_dir),
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _write_run_script(path, command):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export NCCL_P2P_DISABLE=1",
            "export NCCL_IB_DISABLE=1",
            " ".join(command),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a verl GRPO training job from sampled rollout groups.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--work-dir", default="artifacts/verl_grpo_job")
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--n-rollout", type=int, default=4)
    parser.add_argument("--total-epochs", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = prepare_verl_training_job(
        input_jsonl=args.input_jsonl,
        work_dir=args.work_dir,
        model_path=args.model_path,
        n_rollout=args.n_rollout,
        total_epochs=args.total_epochs,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

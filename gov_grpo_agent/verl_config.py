import argparse
import json
from pathlib import Path


def build_verl_grpo_config(
    model_path,
    train_files,
    val_files,
    save_dir,
    reward_path="gov_grpo_agent/verl_reward.py",
    reward_name="compute_score",
    project_name="gov-grpo-agent",
    experiment_name="qwen3_8b_grpo",
    n_rollout=4,
    max_prompt_length=2048,
    max_response_length=512,
    total_epochs=1,
):
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "kl_ctrl": {"kl_coef": 0.03},
        },
        "reward": {
            "custom_reward_function": {
                "path": reward_path,
                "name": reward_name,
            }
        },
        "custom_reward_function": {
            "path": None,
            "name": None,
        },
        "sandbox_fusion": {
            "url": None,
            "max_concurrent": None,
        },
        "reward_model": {
            "num_workers": None,
            "reward_manager": None,
            "reward_loop_source": None,
            "reward_loop_module_path": None,
            "reward_loop_class_name": None,
            "enable": None,
            "enable_resource_pool": None,
            "n_gpus_per_node": None,
            "nnodes": None,
            "reward_kwargs": None,
            "model": {
                "path": None,
            },
            "rollout": {
                "name": None,
            },
        },
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "train_batch_size": 64,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "optim": {"lr": 1e-6},
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": True,
                "kl_loss_coef": 0.03,
            },
            "rollout": {
                "name": "vllm",
                "n": n_rollout,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                "gpu_memory_utilization": 0.75,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
            },
        },
        "trainer": {
            "project_name": project_name,
            "experiment_name": experiment_name,
            "logger": ["console", "tensorboard"],
            "default_local_dir": save_dir,
            "total_epochs": total_epochs,
            "save_freq": 10,
            "test_freq": 10,
            "nnodes": 1,
            "n_gpus_per_node": 8,
        },
    }


def write_verl_grpo_config(
    output_path,
    model_path,
    train_files,
    val_files,
    save_dir,
    reward_path="gov_grpo_agent/verl_reward.py",
    reward_name="compute_score",
    **kwargs,
):
    config = build_verl_grpo_config(
        model_path=model_path,
        train_files=train_files,
        val_files=val_files,
        save_dir=save_dir,
        reward_path=reward_path,
        reward_name=reward_name,
        **kwargs,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_to_yaml(config), encoding="utf-8")
    return config


def build_verl_grpo_command(config):
    override_paths = [
        "algorithm.adv_estimator",
        "algorithm.kl_ctrl.kl_coef",
        "reward.custom_reward_function.path",
        "reward.custom_reward_function.name",
        "data.train_files",
        "data.val_files",
        "data.max_prompt_length",
        "data.max_response_length",
        "data.train_batch_size",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.model.enable_gradient_checkpointing",
        "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.actor.use_kl_loss",
        "actor_rollout_ref.actor.kl_loss_coef",
        "actor_rollout_ref.rollout.name",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.do_sample",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.top_p",
        "actor_rollout_ref.rollout.gpu_memory_utilization",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        "trainer.project_name",
        "trainer.experiment_name",
        "trainer.logger",
        "trainer.default_local_dir",
        "trainer.total_epochs",
        "trainer.save_freq",
        "trainer.test_freq",
        "trainer.nnodes",
        "trainer.n_gpus_per_node",
    ]
    command = [
        "python3",
        "-m",
        "verl.trainer.main_ppo",
    ]
    for path in override_paths:
        command.append(f"{path}={_format_hydra_value(_get_nested(config, path))}")
    return command


def _get_nested(config, path):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def _format_hydra_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def _to_yaml(value, indent=0):
    lines = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
    else:
        lines.append(f"{prefix}{_format_scalar(value)}")
    return "\n".join(lines) + ("\n" if indent == 0 else "")


def _format_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a verl GRPO config for Qwen3-8B.")
    parser.add_argument("--output", default="configs/verl_grpo_qwen3_8b.yaml")
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-files", default="artifacts/verl/train.parquet")
    parser.add_argument("--val-files", default="artifacts/verl/train.parquet")
    parser.add_argument("--save-dir", default="artifacts/verl_grpo_qwen3_8b")
    parser.add_argument("--reward-path", default="gov_grpo_agent/verl_reward.py")
    parser.add_argument("--reward-name", default="compute_score")
    parser.add_argument("--n-rollout", type=int, default=4)
    parser.add_argument("--total-epochs", type=int, default=1)
    args = parser.parse_args(argv)
    config = write_verl_grpo_config(
        output_path=args.output,
        model_path=args.model_path,
        train_files=args.train_files,
        val_files=args.val_files,
        save_dir=args.save_dir,
        reward_path=args.reward_path,
        reward_name=args.reward_name,
        n_rollout=args.n_rollout,
        total_epochs=args.total_epochs,
    )
    print(json.dumps({"config": args.output, "command": build_verl_grpo_command(args.output), "trainer": config["trainer"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

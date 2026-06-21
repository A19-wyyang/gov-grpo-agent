#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python3 -m verl.trainer.main_ppo --config-path configs --config-name verl_grpo_qwen3_8b

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
export TRAIN_FILE="${PROJECT_DIR}/data/processed/train.parquet"
export VAL_FILE="${PROJECT_DIR}/data/processed/validation.parquet"
export TRAIN_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2
export ROLLOUT_N=2
export MAX_MODEL_LEN=2048
export MAX_NUM_SEQS=4
export AGENT_LOOP_WORKERS=4
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.40
export PPO_MAX_TOKEN_LEN_PER_GPU=2048
export TOTAL_EPOCHS=1
export EXPERIMENT_NAME=smoke_grpo

bash "${PROJECT_DIR}/scripts/train_grpo.sh" \
  data.max_response_length=512 \
  trainer.val_before_train=False \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1

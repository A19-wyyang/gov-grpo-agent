#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
export SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-formal/final_adapter}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
export TOTAL_EPOCHS=1
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_formal}"

bash "${PROJECT_DIR}/scripts/train_grpo.sh" \
  data.max_response_length=1024 \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-10}" \
  trainer.save_freq="${SAVE_FREQ:-10}" \
  trainer.test_freq="${TEST_FREQ:-25}"

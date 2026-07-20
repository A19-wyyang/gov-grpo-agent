#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"

# Controlled DAPO Clip-Higher ablation. Keep the negative update bound at
# 0.2 while allowing positive-advantage updates to reach 0.28.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2/final_adapter}"
export TRAIN_FILE="${PROJECT_DIR}/data/processed_v2/train.parquet"
export VAL_FILE="${PROJECT_DIR}/data/processed_v2/validation.parquet"
export TOOL_CONFIG="${PROJECT_DIR}/configs/tools/government_service.yaml"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export FILTER_GROUPS_ENABLE=False
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export LOSS_AGG_MODE=token-mean
export CLIP_RATIO=0.2
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_cliphigher_25step}"

source "${PROJECT_DIR}/scripts/reward_v2_env.sh"
export GOV_JUDGE_ERROR_LOG="${GOV_JUDGE_ERROR_LOG:-${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/judge_errors.jsonl}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.002}"

bash "${PROJECT_DIR}/scripts/run_grpo_with_plots.sh"

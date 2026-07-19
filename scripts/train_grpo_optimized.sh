#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"

# Evidence-driven iteration: preserve B16 x R4 to isolate reward/exploration changes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_25step}"

# Resume-inspired, but deliberately conservative: stronger verifier ownership,
# lower expression weight, and a small entropy increase without changing group size.
source "${PROJECT_DIR}/scripts/reward_v2_env.sh"
export GOV_JUDGE_ERROR_LOG="${GOV_JUDGE_ERROR_LOG:-${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/judge_errors.jsonl}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.002}"

bash "${PROJECT_DIR}/scripts/run_grpo_with_plots.sh"

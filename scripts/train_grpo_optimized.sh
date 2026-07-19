#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"

# Evidence-driven iteration: preserve B16 x R4 to isolate reward/exploration changes.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export TEST_FREQ="${TEST_FREQ:-25}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_25step}"

# Resume-inspired, but deliberately conservative: stronger verifier ownership,
# lower expression weight, and a small entropy increase without changing group size.
export GOV_MISSING_TOOL_PENALTY="${GOV_MISSING_TOOL_PENALTY:-0.45}"
export GOV_MISSING_TOOL_HARD_GATE="${GOV_MISSING_TOOL_HARD_GATE:-1}"
export GOV_INVALID_SLOT_PENALTY="${GOV_INVALID_SLOT_PENALTY:-0.15}"
export GOV_ILLEGAL_ACTION_PENALTY="${GOV_ILLEGAL_ACTION_PENALTY:-0.25}"
export GOV_HARD_FACT_WEIGHT="${GOV_HARD_FACT_WEIGHT:-0.70}"
export GOV_PROCESS_WEIGHT="${GOV_PROCESS_WEIGHT:-0.25}"
export GOV_EXPRESSION_WEIGHT="${GOV_EXPRESSION_WEIGHT:-0.05}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.002}"

bash "${PROJECT_DIR}/scripts/run_grpo_with_plots.sh"

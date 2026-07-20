#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_25step}"
BASELINE_STEP="${BASELINE_STEP:-25}"
CURRICULUM_DIR="${CURRICULUM_DIR:-${PROJECT_DIR}/data/curriculum_v3}"
VALIDATION_METRICS="${VALIDATION_METRICS:-${PROJECT_DIR}/runs/${BASELINE_EXPERIMENT}/validation/${BASELINE_STEP}.jsonl}"
RESCORED_VALIDATION="${CURRICULUM_DIR}/baseline_validation_common_reward.jsonl"

# Controlled sampling ablation: weights come from validation, never test.
source "${PROJECT_DIR}/scripts/reward_v2_env.sh"
"/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/rescore_rollouts.py" \
  --input "${VALIDATION_METRICS}" \
  --cases "${PROJECT_DIR}/data/processed_v2/validation.cases.jsonl" \
  --output "${RESCORED_VALIDATION}"

"/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/build_scenario_curriculum.py" \
  --train-jsonl "${PROJECT_DIR}/data/processed_v2/train.jsonl" \
  --validation-metrics "${RESCORED_VALIDATION}" \
  --metrics-split validation \
  --output-dir "${CURRICULUM_DIR}" \
  --alpha "${CURRICULUM_ALPHA:-1.0}" \
  --max-multiplier "${CURRICULUM_MAX_MULTIPLIER:-2.0}" \
  --max-expansion "${CURRICULUM_MAX_EXPANSION:-1.5}" \
  --target-success "${CURRICULUM_TARGET_SUCCESS:-1.0}" \
  --seed "${CURRICULUM_SEED:-42}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2/final_adapter}"
export TRAIN_FILE="${CURRICULUM_DIR}/train.parquet"
export VAL_FILE="${PROJECT_DIR}/data/processed_v2/validation.parquet"
export TOOL_CONFIG="${PROJECT_DIR}/configs/tools/government_service.yaml"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-25}"
export SAVE_FREQ="${SAVE_FREQ:-5}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_curriculumv3_25step}"

export GOV_JUDGE_ERROR_LOG="${GOV_JUDGE_ERROR_LOG:-${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/judge_errors.jsonl}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.002}"

bash "${PROJECT_DIR}/scripts/run_grpo_with_plots.sh"

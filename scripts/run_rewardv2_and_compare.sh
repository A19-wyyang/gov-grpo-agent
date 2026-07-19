#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step}"
CANDIDATE_EXPERIMENT="${CANDIDATE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_25step}"
TARGET_STEP="${TARGET_STEP:-25}"
COMPARISON_DIR="${COMPARISON_DIR:-${PROJECT_DIR}/results/comparisons/${BASELINE_EXPERIMENT}_vs_${CANDIDATE_EXPERIMENT}}"
PYTHON="${GOVAGENT_PYTHON:-/data/anaconda3/envs/govagent/bin/python}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
mkdir -p "runs/${CANDIDATE_EXPERIMENT}" "${COMPARISON_DIR}"
EXPERIMENT_NAME="${CANDIDATE_EXPERIMENT}" \
  bash scripts/train_grpo_optimized.sh 2>&1 \
  | tee "runs/${CANDIDATE_EXPERIMENT}/train.log"

# A changed reward definition cannot be compared against stored baseline rewards.
# Replay both policies with the exact reward-v2 environment first.
source "${PROJECT_DIR}/scripts/reward_v2_env.sh"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${BASELINE_EXPERIMENT}/validation/${TARGET_STEP}.jsonl" \
  --cases data/processed/validation.cases.jsonl \
  --output "${COMPARISON_DIR}/baseline_common_reward.jsonl"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${CANDIDATE_EXPERIMENT}/validation/${TARGET_STEP}.jsonl" \
  --cases data/processed/validation.cases.jsonl \
  --output "${COMPARISON_DIR}/candidate_common_reward.jsonl"

"${PYTHON}" scripts/compare_grpo_experiments.py \
  --baseline "results/${BASELINE_EXPERIMENT}/validation_metrics.csv" \
  --candidate "results/${CANDIDATE_EXPERIMENT}/validation_metrics.csv" \
  --baseline-jsonl "${COMPARISON_DIR}/baseline_common_reward.jsonl" \
  --candidate-jsonl "${COMPARISON_DIR}/candidate_common_reward.jsonl" \
  --step "${TARGET_STEP}" \
  --baseline-name "${BASELINE_EXPERIMENT}" \
  --candidate-name "${CANDIDATE_EXPERIMENT}" \
  --output-dir "${COMPARISON_DIR}"

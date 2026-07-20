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
  --output "${COMPARISON_DIR}/baseline_common_reward.jsonl" \
  --allow-missing-case-fingerprint
"${PYTHON}" scripts/select_best_grpo_checkpoint.py \
  --validation-dir "runs/${CANDIDATE_EXPERIMENT}/validation" \
  --cases data/processed/validation.cases.jsonl \
  --checkpoint-root "checkpoints/gov_agent_rl/${CANDIDATE_EXPERIMENT}" \
  --output-dir "${COMPARISON_DIR}"
BEST_STEP=$("${PYTHON}" -c \
  "import json; print(json.load(open('${COMPARISON_DIR}/best_checkpoint.json'))['best_step'])")

"${PYTHON}" scripts/compare_grpo_experiments.py \
  --baseline "results/${BASELINE_EXPERIMENT}/validation_metrics.csv" \
  --candidate "results/${CANDIDATE_EXPERIMENT}/validation_metrics.csv" \
  --baseline-jsonl "${COMPARISON_DIR}/baseline_common_reward.jsonl" \
  --candidate-jsonl "${COMPARISON_DIR}/candidate_step_${BEST_STEP}_common_reward.jsonl" \
  --baseline-step "${TARGET_STEP}" \
  --candidate-step "${BEST_STEP}" \
  --baseline-name "${BASELINE_EXPERIMENT}" \
  --candidate-name "${CANDIDATE_EXPERIMENT}" \
  --output-dir "${COMPARISON_DIR}"

# Validation selected the checkpoint. Promotion is decided only on the
# matter-isolated test split to avoid checkpoint-selection bias.
BASELINE_TEST_EXPERIMENT="${BASELINE_EXPERIMENT}_heldout_test"
CANDIDATE_TEST_EXPERIMENT="${CANDIDATE_EXPERIMENT}_best${BEST_STEP}_heldout_test"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
  CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/gov_agent_rl/${BASELINE_EXPERIMENT}/global_step_${TARGET_STEP}" \
  EXPERIMENT_NAME="${BASELINE_TEST_EXPERIMENT}" \
  bash scripts/evaluate_grpo.sh
"${PYTHON}" scripts/export_grpo_metrics.py --experiment "${BASELINE_TEST_EXPERIMENT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}" \
  CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/gov_agent_rl/${CANDIDATE_EXPERIMENT}/global_step_${BEST_STEP}" \
  EXPERIMENT_NAME="${CANDIDATE_TEST_EXPERIMENT}" \
  bash scripts/evaluate_grpo.sh
"${PYTHON}" scripts/export_grpo_metrics.py --experiment "${CANDIDATE_TEST_EXPERIMENT}"

TEST_COMPARISON_DIR="${COMPARISON_DIR}/heldout_test"
mkdir -p "${TEST_COMPARISON_DIR}"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${BASELINE_TEST_EXPERIMENT}/validation/${TARGET_STEP}.jsonl" \
  --cases data/processed/test.cases.jsonl \
  --output "${TEST_COMPARISON_DIR}/baseline_common_reward.jsonl" \
  --allow-missing-case-fingerprint
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${CANDIDATE_TEST_EXPERIMENT}/validation/${BEST_STEP}.jsonl" \
  --cases data/processed/test.cases.jsonl \
  --output "${TEST_COMPARISON_DIR}/candidate_common_reward.jsonl" \
  --allow-missing-case-fingerprint
"${PYTHON}" scripts/compare_grpo_experiments.py \
  --baseline "results/${BASELINE_TEST_EXPERIMENT}/validation_metrics.csv" \
  --candidate "results/${CANDIDATE_TEST_EXPERIMENT}/validation_metrics.csv" \
  --baseline-jsonl "${TEST_COMPARISON_DIR}/baseline_common_reward.jsonl" \
  --candidate-jsonl "${TEST_COMPARISON_DIR}/candidate_common_reward.jsonl" \
  --baseline-step "${TARGET_STEP}" \
  --candidate-step "${BEST_STEP}" \
  --baseline-name "${BASELINE_TEST_EXPERIMENT}" \
  --candidate-name "${CANDIDATE_TEST_EXPERIMENT}" \
  --output-dir "${TEST_COMPARISON_DIR}"
"${PYTHON}" scripts/decide_grpo_promotion.py \
  --comparison "${TEST_COMPARISON_DIR}/comparison.json" \
  --output "${TEST_COMPARISON_DIR}/promotion_decision.json"

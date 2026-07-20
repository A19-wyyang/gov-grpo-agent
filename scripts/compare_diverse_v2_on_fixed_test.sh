#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
PYTHON="${GOVAGENT_PYTHON:-/data/anaconda3/envs/govagent/bin/python}"
LEGACY_EXPERIMENT="${LEGACY_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_25step}"
DIVERSE_EXPERIMENT="${DIVERSE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_25step}"
LEGACY_BEST_JSON="${LEGACY_BEST_JSON:-${PROJECT_DIR}/results/comparisons/qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step_vs_${LEGACY_EXPERIMENT}/best_checkpoint.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/comparisons/${LEGACY_EXPERIMENT}_vs_${DIVERSE_EXPERIMENT}_fixed_v2_test}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/reward_v2_env.sh"

# Select the diverse-data checkpoint only on diverse_v2 validation.
"${PYTHON}" scripts/select_best_grpo_checkpoint.py \
  --validation-dir "runs/${DIVERSE_EXPERIMENT}/validation" \
  --cases data/processed_v2/validation.cases.jsonl \
  --checkpoint-root "checkpoints/gov_agent_rl/${DIVERSE_EXPERIMENT}" \
  --output-dir "${OUTPUT_DIR}/diverse_validation_selection"

LEGACY_CHECKPOINT=$("${PYTHON}" -c \
  "import json; print(json.load(open('${LEGACY_BEST_JSON}'))['best_checkpoint'])")
DIVERSE_CHECKPOINT=$("${PYTHON}" -c \
  "import json; print(json.load(open('${OUTPUT_DIR}/diverse_validation_selection/best_checkpoint.json'))['best_checkpoint'])")
LEGACY_STEP="${LEGACY_CHECKPOINT##*global_step_}"
DIVERSE_STEP="${DIVERSE_CHECKPOINT##*global_step_}"

# The fixed v2 test is evaluated only after validation has selected both models.
LEGACY_TEST_EXPERIMENT="${LEGACY_EXPERIMENT}_fixed_v2_test"
DIVERSE_TEST_EXPERIMENT="${DIVERSE_EXPERIMENT}_fixed_v2_test"
EVAL_DATA_DIR="${PROJECT_DIR}/data/processed_v2" \
  EVAL_ROLLOUT_N=4 \
  CHECKPOINT_PATH="${LEGACY_CHECKPOINT}" \
  SFT_ADAPTER="${PROJECT_DIR}/outputs/sft-qwen3-8b-formal/final_adapter" \
  EXPERIMENT_NAME="${LEGACY_TEST_EXPERIMENT}" \
  bash scripts/evaluate_grpo.sh
"${PYTHON}" scripts/export_grpo_metrics.py --experiment "${LEGACY_TEST_EXPERIMENT}"

EVAL_DATA_DIR="${PROJECT_DIR}/data/processed_v2" \
  EVAL_ROLLOUT_N=4 \
  CHECKPOINT_PATH="${DIVERSE_CHECKPOINT}" \
  SFT_ADAPTER="${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2/final_adapter" \
  EXPERIMENT_NAME="${DIVERSE_TEST_EXPERIMENT}" \
  bash scripts/evaluate_grpo.sh
"${PYTHON}" scripts/export_grpo_metrics.py --experiment "${DIVERSE_TEST_EXPERIMENT}"

mkdir -p "${OUTPUT_DIR}/fixed_v2_test"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${LEGACY_TEST_EXPERIMENT}/validation/${LEGACY_STEP}.jsonl" \
  --cases data/processed_v2/test.cases.jsonl \
  --output "${OUTPUT_DIR}/fixed_v2_test/legacy_common_reward.jsonl"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${DIVERSE_TEST_EXPERIMENT}/validation/${DIVERSE_STEP}.jsonl" \
  --cases data/processed_v2/test.cases.jsonl \
  --output "${OUTPUT_DIR}/fixed_v2_test/diverse_common_reward.jsonl"

"${PYTHON}" scripts/compare_grpo_experiments.py \
  --baseline "results/${LEGACY_TEST_EXPERIMENT}/validation_metrics.csv" \
  --candidate "results/${DIVERSE_TEST_EXPERIMENT}/validation_metrics.csv" \
  --baseline-jsonl "${OUTPUT_DIR}/fixed_v2_test/legacy_common_reward.jsonl" \
  --candidate-jsonl "${OUTPUT_DIR}/fixed_v2_test/diverse_common_reward.jsonl" \
  --baseline-step "${LEGACY_STEP}" \
  --candidate-step "${DIVERSE_STEP}" \
  --baseline-name "${LEGACY_TEST_EXPERIMENT}" \
  --candidate-name "${DIVERSE_TEST_EXPERIMENT}" \
  --output-dir "${OUTPUT_DIR}/fixed_v2_test"
"${PYTHON}" scripts/decide_grpo_promotion.py \
  --comparison "${OUTPUT_DIR}/fixed_v2_test/comparison.json" \
  --output "${OUTPUT_DIR}/fixed_v2_test/promotion_decision.json"

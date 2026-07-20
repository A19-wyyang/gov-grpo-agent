#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
PYTHON="${GOVAGENT_PYTHON:-/data/anaconda3/envs/govagent/bin/python}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_25step}"
CANDIDATE_EXPERIMENT="${CANDIDATE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_curriculumv3_25step}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/comparisons/${BASELINE_EXPERIMENT}_vs_${CANDIDATE_EXPERIMENT}_fixed_v2_test}"
BASELINE_SFT_ADAPTER="${BASELINE_SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2/final_adapter}"
CANDIDATE_SFT_ADAPTER="${CANDIDATE_SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2/final_adapter}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
source "${PROJECT_DIR}/scripts/reward_v2_env.sh"

# Both checkpoints are selected independently using the same v2 validation set.
for side in baseline candidate; do
  if [[ "${side}" == "baseline" ]]; then
    experiment="${BASELINE_EXPERIMENT}"
  else
    experiment="${CANDIDATE_EXPERIMENT}"
  fi
  "${PYTHON}" scripts/select_best_grpo_checkpoint.py \
    --validation-dir "runs/${experiment}/validation" \
    --cases data/processed_v2/validation.cases.jsonl \
    --checkpoint-root "checkpoints/gov_agent_rl/${experiment}" \
    --output-dir "${OUTPUT_DIR}/${side}_validation_selection"
done

BASELINE_CHECKPOINT=$("${PYTHON}" -c \
  "import json; print(json.load(open('${OUTPUT_DIR}/baseline_validation_selection/best_checkpoint.json'))['best_checkpoint'])")
CANDIDATE_CHECKPOINT=$("${PYTHON}" -c \
  "import json; print(json.load(open('${OUTPUT_DIR}/candidate_validation_selection/best_checkpoint.json'))['best_checkpoint'])")
BASELINE_STEP="${BASELINE_CHECKPOINT##*global_step_}"
CANDIDATE_STEP="${CANDIDATE_CHECKPOINT##*global_step_}"
BASELINE_TEST_EXPERIMENT="${BASELINE_EXPERIMENT}_fixed_v2_ablation_test"
CANDIDATE_TEST_EXPERIMENT="${CANDIDATE_EXPERIMENT}_fixed_v2_ablation_test"

for side in baseline candidate; do
  if [[ "${side}" == "baseline" ]]; then
    checkpoint="${BASELINE_CHECKPOINT}"
    experiment="${BASELINE_TEST_EXPERIMENT}"
    adapter="${BASELINE_SFT_ADAPTER}"
  else
    checkpoint="${CANDIDATE_CHECKPOINT}"
    experiment="${CANDIDATE_TEST_EXPERIMENT}"
    adapter="${CANDIDATE_SFT_ADAPTER}"
  fi
  EVAL_DATA_DIR="${PROJECT_DIR}/data/processed_v2" \
    EVAL_ROLLOUT_N=4 \
    CHECKPOINT_PATH="${checkpoint}" \
    SFT_ADAPTER="${adapter}" \
    EXPERIMENT_NAME="${experiment}" \
    bash scripts/evaluate_grpo.sh
  "${PYTHON}" scripts/export_grpo_metrics.py --experiment "${experiment}"
done

mkdir -p "${OUTPUT_DIR}/fixed_v2_test"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${BASELINE_TEST_EXPERIMENT}/validation/${BASELINE_STEP}.jsonl" \
  --cases data/processed_v2/test.cases.jsonl \
  --output "${OUTPUT_DIR}/fixed_v2_test/baseline_common_reward.jsonl"
"${PYTHON}" scripts/rescore_rollouts.py \
  --input "runs/${CANDIDATE_TEST_EXPERIMENT}/validation/${CANDIDATE_STEP}.jsonl" \
  --cases data/processed_v2/test.cases.jsonl \
  --output "${OUTPUT_DIR}/fixed_v2_test/candidate_common_reward.jsonl"

"${PYTHON}" scripts/compare_grpo_experiments.py \
  --baseline "results/${BASELINE_TEST_EXPERIMENT}/validation_metrics.csv" \
  --candidate "results/${CANDIDATE_TEST_EXPERIMENT}/validation_metrics.csv" \
  --baseline-jsonl "${OUTPUT_DIR}/fixed_v2_test/baseline_common_reward.jsonl" \
  --candidate-jsonl "${OUTPUT_DIR}/fixed_v2_test/candidate_common_reward.jsonl" \
  --baseline-step "${BASELINE_STEP}" \
  --candidate-step "${CANDIDATE_STEP}" \
  --baseline-name "${BASELINE_TEST_EXPERIMENT}" \
  --candidate-name "${CANDIDATE_TEST_EXPERIMENT}" \
  --output-dir "${OUTPUT_DIR}/fixed_v2_test"
"${PYTHON}" scripts/decide_grpo_promotion.py \
  --comparison "${OUTPUT_DIR}/fixed_v2_test/comparison.json" \
  --output "${OUTPUT_DIR}/fixed_v2_test/promotion_decision.json"

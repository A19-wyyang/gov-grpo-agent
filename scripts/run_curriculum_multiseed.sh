#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
PYTHON="${GOVAGENT_PYTHON:-/data/anaconda3/envs/govagent/bin/python}"
SEEDS_TEXT="${SEEDS:-42 43 44}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CANDIDATE="${RUN_CANDIDATE:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
read -r -a SEED_ARRAY <<< "${SEEDS_TEXT}"

if [[ "${#SEED_ARRAY[@]}" -lt 3 ]]; then
  echo "Formal multi-seed evidence requires at least three seeds" >&2
  exit 2
fi

comparison_args=()
for seed in "${SEED_ARRAY[@]}"; do
  baseline="qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_seed${seed}"
  candidate="qwen3_8b_gov_agent_grpo_b16r4_rewardv2_curriculumv3_seed${seed}"
  comparison_dir="${PROJECT_DIR}/results/comparisons/${baseline}_vs_${candidate}_fixed_v2_test"

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    TRAIN_SEED="${seed}" \
      EXPERIMENT_NAME="${baseline}" \
      bash "${PROJECT_DIR}/scripts/train_grpo_diverse_v2.sh"
  fi
  if [[ "${RUN_CANDIDATE}" == "1" ]]; then
    TRAIN_SEED="${seed}" \
      CURRICULUM_SEED="${seed}" \
      CURRICULUM_DIR="${PROJECT_DIR}/data/curriculum_v3_seed${seed}" \
      BASELINE_EXPERIMENT="${baseline}" \
      EXPERIMENT_NAME="${candidate}" \
      bash "${PROJECT_DIR}/scripts/train_grpo_curriculum_v3.sh"
  fi
  if [[ "${RUN_COMPARISON}" == "1" ]]; then
    BASELINE_EXPERIMENT="${baseline}" \
      CANDIDATE_EXPERIMENT="${candidate}" \
      OUTPUT_DIR="${comparison_dir}" \
      bash "${PROJECT_DIR}/scripts/compare_curriculum_v3_on_fixed_test.sh"
  fi
  comparison_args+=(
    --comparison
    "${seed}=${comparison_dir}/fixed_v2_test/comparison.json"
  )
done

"${PYTHON}" "${PROJECT_DIR}/scripts/aggregate_seed_comparisons.py" \
  "${comparison_args[@]}" \
  --output-dir "${PROJECT_DIR}/results/comparisons/curriculum_v3_multiseed"

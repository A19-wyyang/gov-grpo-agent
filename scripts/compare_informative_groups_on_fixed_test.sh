#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_25step}"
CANDIDATE_EXPERIMENT="${CANDIDATE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_informative_groups_25step}"

BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT}" \
  CANDIDATE_EXPERIMENT="${CANDIDATE_EXPERIMENT}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/comparisons/${BASELINE_EXPERIMENT}_vs_${CANDIDATE_EXPERIMENT}_fixed_v2_test}" \
  bash "${PROJECT_DIR}/scripts/compare_curriculum_v3_on_fixed_test.sh"

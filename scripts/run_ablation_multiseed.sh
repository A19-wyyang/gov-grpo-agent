#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
PYTHON="${GOVAGENT_PYTHON:-/data/anaconda3/envs/govagent/bin/python}"
SEEDS_TEXT="${SEEDS:-42 43 44}"
BASELINE_TRAIN_SCRIPT="${BASELINE_TRAIN_SCRIPT:-${PROJECT_DIR}/scripts/train_grpo_diverse_v2.sh}"
CANDIDATE_TRAIN_SCRIPT="${CANDIDATE_TRAIN_SCRIPT:?set CANDIDATE_TRAIN_SCRIPT}"
CANDIDATE_SLUG="${CANDIDATE_SLUG:?set CANDIDATE_SLUG}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${PROJECT_DIR}/scripts/compare_curriculum_v3_on_fixed_test.sh}"
SCREENING_DECISION_FILE="${SCREENING_DECISION_FILE:?set SCREENING_DECISION_FILE to the single-seed promotion_decision.json}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CANDIDATE="${RUN_CANDIDATE:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
read -r -a SEED_ARRAY <<< "${SEEDS_TEXT}"

if [[ "${#SEED_ARRAY[@]}" -lt 3 ]]; then
  echo "Formal multi-seed evidence requires at least three seeds" >&2
  exit 2
fi
if [[ ! "${CANDIDATE_SLUG}" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  echo "CANDIDATE_SLUG must contain only lowercase letters, digits, _ or -" >&2
  exit 2
fi
for script in "${BASELINE_TRAIN_SCRIPT}" "${CANDIDATE_TRAIN_SCRIPT}" "${COMPARE_SCRIPT}"; do
  if [[ ! -f "${script}" ]]; then
    echo "Missing script: ${script}" >&2
    exit 2
  fi
done
if [[ ! -f "${SCREENING_DECISION_FILE}" ]]; then
  echo "Missing single-seed screening decision: ${SCREENING_DECISION_FILE}" >&2
  exit 2
fi

screening_decision=$("${PYTHON}" -c \
  "import json, sys; print(json.load(open(sys.argv[1]))['decision'])" \
  "${SCREENING_DECISION_FILE}")
if [[ "${screening_decision}" == "reject" || "${screening_decision}" == "invalid" ]]; then
  echo "Candidate failed single-seed screening: ${screening_decision}" >&2
  exit 3
fi
if [[ "${screening_decision}" != "promote" && "${screening_decision}" != "needs_more_evidence" ]]; then
  echo "Unsupported screening decision: ${screening_decision}" >&2
  exit 3
fi

comparison_args=()
for seed in "${SEED_ARRAY[@]}"; do
  baseline="qwen3_8b_gov_agent_grpo_b16r4_rewardv2_diversev2_seed${seed}"
  candidate="qwen3_8b_gov_agent_grpo_b16r4_rewardv2_${CANDIDATE_SLUG}_seed${seed}"
  comparison_dir="${PROJECT_DIR}/results/comparisons/${baseline}_vs_${candidate}_fixed_v2_test"

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    TRAIN_SEED="${seed}" \
      ROLLOUT_SEED="${seed}" \
      EXPERIMENT_NAME="${baseline}" \
      bash "${BASELINE_TRAIN_SCRIPT}"
  fi
  if [[ "${RUN_CANDIDATE}" == "1" ]]; then
    TRAIN_SEED="${seed}" \
      ROLLOUT_SEED="${seed}" \
      EXPERIMENT_NAME="${candidate}" \
      bash "${CANDIDATE_TRAIN_SCRIPT}"
  fi
  if [[ "${RUN_COMPARISON}" == "1" ]]; then
    BASELINE_EXPERIMENT="${baseline}" \
      CANDIDATE_EXPERIMENT="${candidate}" \
      OUTPUT_DIR="${comparison_dir}" \
      bash "${COMPARE_SCRIPT}"
  fi
  comparison_args+=(
    --comparison
    "${seed}=${comparison_dir}/fixed_v2_test/comparison.json"
  )
done

"${PYTHON}" "${PROJECT_DIR}/scripts/aggregate_seed_comparisons.py" \
  "${comparison_args[@]}" \
  --title "${CANDIDATE_SLUG} cross-seed A/B deltas (95% t interval)" \
  --output-dir "${PROJECT_DIR}/results/comparisons/${CANDIDATE_SLUG}_multiseed"

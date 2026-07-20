#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step}"
RUN_DIR="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}"

mkdir -p "${RUN_DIR}"
status=0
bash "${PROJECT_DIR}/scripts/train_grpo_formal.sh" || status=$?

if [[ "${status}" -eq 0 ]]; then
  "/data/anaconda3/envs/govagent/bin/python" \
    "${PROJECT_DIR}/scripts/export_grpo_metrics.py" \
    --experiment "${EXPERIMENT_NAME}" \
    --require-critical-metrics
  "/data/anaconda3/envs/govagent/bin/python" \
    "${PROJECT_DIR}/scripts/analyze_rollout_budget.py" \
    --rollout-dir "${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/rollouts" \
    --output-dir "${PROJECT_DIR}/results/${EXPERIMENT_NAME}" \
    --targets 4 8 16
else
  "/data/anaconda3/envs/govagent/bin/python" \
    "${PROJECT_DIR}/scripts/export_grpo_metrics.py" \
    --experiment "${EXPERIMENT_NAME}" || true
fi

exit "${status}"

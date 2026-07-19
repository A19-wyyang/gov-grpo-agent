#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step}"
RUN_DIR="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}"

mkdir -p "${RUN_DIR}"
status=0
bash "${PROJECT_DIR}/scripts/train_grpo_formal.sh" || status=$?

"/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/export_grpo_metrics.py" \
  --experiment "${EXPERIMENT_NAME}" || true

exit "${status}"

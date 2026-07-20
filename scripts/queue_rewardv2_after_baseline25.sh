#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step}"
CANDIDATE_EXPERIMENT="${CANDIDATE_EXPERIMENT:-qwen3_8b_gov_agent_grpo_b16r4_rewardv2_25step}"
BASELINE_SESSION="${BASELINE_SESSION:-govgrpo_b16r4}"
CANDIDATE_SESSION="${CANDIDATE_SESSION:-govgrpo_rewardv2}"
TARGET_STEP="${TARGET_STEP:-25}"

validation_file="${PROJECT_DIR}/runs/${BASELINE_EXPERIMENT}/validation/${TARGET_STEP}.jsonl"
checkpoint_dir="${PROJECT_DIR}/checkpoints/gov_agent_rl/${BASELINE_EXPERIMENT}/global_step_${TARGET_STEP}"
comparison_dir="${PROJECT_DIR}/results/comparisons/${BASELINE_EXPERIMENT}_vs_${CANDIDATE_EXPERIMENT}"

echo "Waiting for baseline validation and checkpoint at step ${TARGET_STEP}..."
until "/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/check_grpo_snapshot.py" \
  --validation "${validation_file}" \
  --checkpoint "${checkpoint_dir}" \
  --step "${TARGET_STEP}" \
  --expected-cases "${EXPECTED_VALIDATION_CASES:-200}" \
  --world-size "${CHECKPOINT_WORLD_SIZE:-2}" \
  --min-age-seconds "${SNAPSHOT_SETTLE_SECONDS:-30}" >/dev/null 2>&1; do
  if ! tmux has-session -t "${BASELINE_SESSION}" 2>/dev/null; then
    echo "Baseline session ended before step ${TARGET_STEP}." >&2
    exit 1
  fi
  sleep 30
done

echo "Baseline step ${TARGET_STEP} is durable; exporting its metrics."
"/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/check_grpo_snapshot.py" \
  --validation "${validation_file}" \
  --checkpoint "${checkpoint_dir}" \
  --step "${TARGET_STEP}" \
  --expected-cases "${EXPECTED_VALIDATION_CASES:-200}" \
  --world-size "${CHECKPOINT_WORLD_SIZE:-2}" \
  --min-age-seconds "${SNAPSHOT_SETTLE_SECONDS:-30}"
"/data/anaconda3/envs/govagent/bin/python" \
  "${PROJECT_DIR}/scripts/export_grpo_metrics.py" \
  --experiment "${BASELINE_EXPERIMENT}"

# Stop only this project's baseline after its comparable snapshot is durable.
tmux send-keys -t "${BASELINE_SESSION}" C-c
for _ in $(seq 1 20); do
  if ! pgrep -af "trainer.experiment_name=${BASELINE_EXPERIMENT}" >/dev/null; then
    break
  fi
  sleep 15
done
if pgrep -af "trainer.experiment_name=${BASELINE_EXPERIMENT}" >/dev/null; then
  echo "Baseline trainer did not stop cleanly; candidate was not started." >&2
  exit 1
fi

echo "Starting candidate experiment ${CANDIDATE_EXPERIMENT}."
tmux new-session -d -s "${CANDIDATE_SESSION}" \
  "cd '${PROJECT_DIR}' && PROJECT_DIR='${PROJECT_DIR}' BASELINE_EXPERIMENT='${BASELINE_EXPERIMENT}' CANDIDATE_EXPERIMENT='${CANDIDATE_EXPERIMENT}' TARGET_STEP='${TARGET_STEP}' COMPARISON_DIR='${comparison_dir}' bash scripts/run_rewardv2_and_compare.sh"

echo "Candidate queued in tmux session ${CANDIDATE_SESSION}."

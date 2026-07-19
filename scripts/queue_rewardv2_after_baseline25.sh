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
while [[ ! -s "${validation_file}" || ! -d "${checkpoint_dir}" ]]; do
  if ! tmux has-session -t "${BASELINE_SESSION}" 2>/dev/null; then
    echo "Baseline session ended before step ${TARGET_STEP}." >&2
    exit 1
  fi
  sleep 30
done

echo "Baseline step ${TARGET_STEP} is durable; exporting its metrics."
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
  "cd '${PROJECT_DIR}' && mkdir -p 'runs/${CANDIDATE_EXPERIMENT}' && EXPERIMENT_NAME='${CANDIDATE_EXPERIMENT}' bash scripts/train_grpo_optimized.sh 2>&1 | tee 'runs/${CANDIDATE_EXPERIMENT}/train.log'; status=\${PIPESTATUS[0]}; if [[ \${status} -eq 0 ]]; then '/data/anaconda3/envs/govagent/bin/python' scripts/compare_grpo_experiments.py --baseline 'results/${BASELINE_EXPERIMENT}/validation_metrics.csv' --candidate 'results/${CANDIDATE_EXPERIMENT}/validation_metrics.csv' --baseline-jsonl 'runs/${BASELINE_EXPERIMENT}/validation/${TARGET_STEP}.jsonl' --candidate-jsonl 'runs/${CANDIDATE_EXPERIMENT}/validation/${TARGET_STEP}.jsonl' --step '${TARGET_STEP}' --baseline-name '${BASELINE_EXPERIMENT}' --candidate-name '${CANDIDATE_EXPERIMENT}' --output-dir '${comparison_dir}'; fi; exit \${status}"

echo "Candidate queued in tmux session ${CANDIDATE_SESSION}."

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/checkpoints/gov_agent_rl/qwen3_8b_gov_agent_grpo_formal/global_step_10}"
export SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b-formal/final_adapter}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_DIR}/data/processed}"
export TRAIN_FILE="${TRAIN_FILE:-${EVAL_DATA_DIR}/train.parquet}"
export VAL_FILE="${VAL_FILE:-${EVAL_DATA_DIR}/test.parquet}"
EVAL_CASES_FILE="${EVAL_CASES_FILE:-${EVAL_DATA_DIR}/test.cases.jsonl}"
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export EVAL_ROLLOUT_N="${EVAL_ROLLOUT_N:-4}"
export EVAL_SEED="${EVAL_SEED:-20260719}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-${EVAL_SEED}}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
export EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-True}"
export ROLLOUT_N="${ROLLOUT_N:-${EVAL_ROLLOUT_N}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
export TOTAL_EPOCHS=1
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo_test}"

bash "${PROJECT_DIR}/scripts/train_grpo.sh" \
  data.max_response_length=1024 \
  'actor_rollout_ref.actor.checkpoint.load_contents=["model"]' \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${CHECKPOINT_PATH}" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1

CHECKPOINT_STEP="${CHECKPOINT_PATH##*global_step_}"
EVAL_JSONL="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/validation/${CHECKPOINT_STEP}.jsonl"
python "${PROJECT_DIR}/scripts/check_rollout_coverage.py" \
  --rollouts "${EVAL_JSONL}" \
  --cases "${EVAL_CASES_FILE}" \
  --rollouts-per-case "${EVAL_ROLLOUT_N}" \
  --step "${CHECKPOINT_STEP}"
python -m gov_agent_rl evaluate \
  --input "${EVAL_JSONL}" \
  --out "${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/metrics.json"

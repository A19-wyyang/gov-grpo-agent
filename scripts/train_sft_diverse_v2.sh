#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft-qwen3-8b-diverse-v2}"

bash "${PROJECT_DIR}/scripts/train_sft.sh" \
  --train "${PROJECT_DIR}/data/processed_v2/train.sft.jsonl" \
  --validation "${PROJECT_DIR}/data/processed_v2/validation.sft.jsonl" \
  "$@"

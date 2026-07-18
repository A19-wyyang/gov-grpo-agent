#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft-qwen3-8b}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Consumer RTX 4090 pairs do not support NCCL P2P/IB fast paths.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cd "${PROJECT_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_sft.py \
  --model "${MODEL_PATH}" \
  --train data/processed/train.sft.jsonl \
  --validation data/processed/validation.sft.jsonl \
  --output "${OUTPUT_DIR}" \
  "$@"

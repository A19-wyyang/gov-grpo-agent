#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
CANDIDATE_TRAIN_SCRIPT="${PROJECT_DIR}/scripts/train_grpo_informative_groups.sh" \
  CANDIDATE_SLUG="informative_groups" \
  bash "${PROJECT_DIR}/scripts/run_ablation_multiseed.sh"

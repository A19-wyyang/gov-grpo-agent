#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/code_repos/ywy/Project}"
JUDGE_ENV_FILE="${GOV_JUDGE_ENV_FILE:-${PROJECT_DIR}/.env.judge}"
if [[ -f "${JUDGE_ENV_FILE}" ]]; then
  set -a
  source "${JUDGE_ENV_FILE}"
  set +a
fi
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECT_DIR}/outputs/sft-qwen3-8b/final_adapter}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_DIR}/data/processed/train.parquet}"
VAL_FILE="${VAL_FILE:-${PROJECT_DIR}/data/processed/validation.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-${PROJECT_DIR}/configs/tools/government_service.yaml}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
AGENT_LOOP_WORKERS="${AGENT_LOOP_WORKERS:-8}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_gov_agent_grpo}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
CLIP_RATIO="${CLIP_RATIO:-0.2}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-${CLIP_RATIO}}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-${CLIP_RATIO}}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.02}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
TRAIN_SEED="${TRAIN_SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-${TRAIN_SEED}}"
EVAL_ROLLOUT_N="${EVAL_ROLLOUT_N:-${ROLLOUT_N}}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
EVAL_TOP_P="${EVAL_TOP_P:-0.95}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-True}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-score}"
FILTER_MAX_GEN_BATCHES="${FILTER_MAX_GEN_BATCHES:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONHASHSEED="${TRAIN_SEED}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_USE_V1=1
# The server training workflow pre-caches the base model. Avoid repeated
# Hub HEAD requests (and multi-minute retry stalls) on restricted networks.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${PROJECT_DIR}"
python scripts/write_run_manifest.py \
  --project-dir "${PROJECT_DIR}" \
  --experiment "${EXPERIMENT_NAME}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --model "${MODEL_PATH}" \
  --sft-adapter "${SFT_ADAPTER}" \
  --tool-config "${TOOL_CONFIG}" \
  --rollout-n "${ROLLOUT_N}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --ppo-mini-batch-size "${PPO_MINI_BATCH_SIZE}" \
  --actor-lr "${ACTOR_LR}" \
  --clip-ratio "${CLIP_RATIO}" \
  --clip-ratio-low "${CLIP_RATIO_LOW}" \
  --clip-ratio-high "${CLIP_RATIO_HIGH}" \
  --kl-loss-coef "${KL_LOSS_COEF}" \
  --entropy-coeff "${ENTROPY_COEFF}" \
  --seed "${TRAIN_SEED}" \
  --rollout-seed "${ROLLOUT_SEED}" \
  --eval-rollout-n "${EVAL_ROLLOUT_N}" \
  --eval-temperature "${EVAL_TEMPERATURE}" \
  --eval-top-p "${EVAL_TOP_P}" \
  --eval-do-sample "${EVAL_DO_SAMPLE}" \
  --loss-agg-mode "${LOSS_AGG_MODE}" \
  --gen-batch-size "${GEN_BATCH_SIZE}" \
  --filter-groups-enable "${FILTER_GROUPS_ENABLE}" \
  --filter-groups-metric "${FILTER_GROUPS_METRIC}" \
  --filter-max-gen-batches "${FILTER_MAX_GEN_BATCHES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --total-epochs "${TOTAL_EPOCHS}" \
  --trainer-overrides "$*" \
  --output "${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/run_manifest.json"
python scripts/patch_verl_sdpa.py
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  algorithm.filter_groups.max_num_gen_batches="${FILTER_MAX_GEN_BATCHES}" \
  data.seed="${TRAIN_SEED}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_adapter_path="${SFT_ADAPTER}" \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}" \
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.data_loader_seed="${TRAIN_SEED}" \
  actor_rollout_ref.actor.fsdp_config.seed="${TRAIN_SEED}" \
  actor_rollout_ref.actor.fsdp_config.offload_policy=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.logprobs_mode=null \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.seed="${ROLLOUT_SEED}" \
  actor_rollout_ref.rollout.val_kwargs.n="${EVAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${EVAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${EVAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${EVAL_DO_SAMPLE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS}" \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.seed="${TRAIN_SEED}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  custom_reward_function.path="${PROJECT_DIR}/src/gov_agent_rl/verl_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=gov_agent_rl \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.rollout_data_dir="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/rollouts" \
  trainer.validation_data_dir="${PROJECT_DIR}/runs/${EXPERIMENT_NAME}/validation" \
  "$@"

# verl GRPO Training Workflow

This project uses verl for the GRPO training stage after sampled rollout data has been generated.

## 1. Prepare sampled GRPO data

Run sampled multi-GPU rollout first. Greedy rollout usually produces zero within-group variance and cannot train GRPO.

```bash
python -m gov_grpo_agent.parallel_rollout \
  --gpus 0,1,2,3,4,5,6,7 \
  --total-cases 200 \
  --model-name-or-path Qwen/Qwen3-8B \
  --adapter-path artifacts/qwen3_8b_sft_lora \
  --output-root artifacts/parallel_model_rollout_sampled \
  --rollout-group-size 4 \
  --max-turns 8 \
  --do-sample \
  --temperature 1.1 \
  --top-p 0.92
```

Merge shards and prepare grouped GRPO JSONL:

```bash
python -m gov_grpo_agent.merge_rollout_shards \
  --input-root artifacts/parallel_model_rollout_sampled \
  --output-dir artifacts/model_rollout_sampled_merged

python -m gov_grpo_agent.prepare_grpo \
  --input artifacts/model_rollout_sampled_merged/model_grpo_groups.json \
  --output artifacts/grpo_train/qwen3_grpo_train_sampled.jsonl \
  --report artifacts/grpo_train/qwen3_grpo_report_sampled.json
```

Do not continue if `usable_groups` is `0`.

## 2. Prepare a verl training job

```bash
python -m gov_grpo_agent.train_grpo_verl \
  --input-jsonl artifacts/grpo_train/qwen3_grpo_train_sampled.jsonl \
  --work-dir artifacts/verl_grpo_qwen3_8b \
  --model-path Qwen/Qwen3-8B \
  --n-rollout 4 \
  --total-epochs 1 \
  --gpus 4,5,6,7
```

`--gpus` selects physical devices and automatically sets
`trainer.n_gpus_per_node` to the number of selected GPUs. On the current
mixed-memory server, GPUs 0-3 have 24 GB while GPUs 4-7 have 48 GB, so GRPO
training should use `--gpus 4,5,6,7`.

This writes:

- `artifacts/verl_grpo_qwen3_8b/data/train.parquet`
- `artifacts/verl_grpo_qwen3_8b/data/data_report.json`
- `artifacts/verl_grpo_qwen3_8b/configs/verl_grpo_qwen3_8b.yaml`
- `artifacts/verl_grpo_qwen3_8b/run_verl_grpo.sh`
- `artifacts/verl_grpo_qwen3_8b/manifest.json`

## 3. Run verl

Install server dependencies in the training environment:

```bash
pip install -U verl vllm pyarrow tensorboard
```

Then run:

```bash
bash artifacts/verl_grpo_qwen3_8b/run_verl_grpo.sh
```

The generated verl config uses:

- `algorithm.adv_estimator: grpo`
- `actor_rollout_ref.rollout.n: 4`
- `reward.custom_reward_function.path: gov_grpo_agent/verl_reward.py`
- `trainer.logger: [console, tensorboard]`

The generated shell script loads verl's packaged `ppo_trainer` configuration and
passes these values as Hydra overrides. The YAML under the job directory is a
human-readable record of the selected overrides; it is not used as a replacement
for verl's complete configuration schema.

verl also supports changing the logger to W&B if the server can log in to Weights & Biases.

## 4. Visualize metrics

Build a local HTML and CSV report from JSON/JSONL metric files:

```bash
python -m gov_grpo_agent.metrics_report \
  --metrics artifacts/model_rollout_sampled_merged/model_metrics.json artifacts/verl_grpo_qwen3_8b/data/data_report.json \
  --grpo-report artifacts/grpo_train/qwen3_grpo_report_sampled.json \
  --output-dir artifacts/reports/qwen3_grpo \
  --title "Qwen3-8B GRPO Metrics"
```

Open:

```bash
artifacts/reports/qwen3_grpo/index.html
```

Export JSON/JSONL metrics to TensorBoard event files:

```bash
python -m gov_grpo_agent.tensorboard_export \
  --metrics artifacts/model_rollout_sampled_merged/model_metrics.json artifacts/verl_grpo_qwen3_8b/data/data_report.json \
  --log-dir artifacts/tensorboard/qwen3_grpo

tensorboard --logdir artifacts/tensorboard/qwen3_grpo --host 0.0.0.0 --port 6006
```

## 5. Metrics to track

Training and data quality:

- `avg_reward`
- `best_reward`
- `usable_group_rate`
- `low_variance_groups`
- `actor/kl_loss` from verl console or W&B logs
- `actor/entropy` from verl console or W&B logs
- `actor/pg_loss` from verl console or W&B logs

Agent evaluation:

- `success_at_1`
- `required_tool_recall`
- `final_decision_accuracy`
- `material_check_call_rate`
- `premature_submit_rate`
- `missing_tool_rate`
- `invalid_action_rate`

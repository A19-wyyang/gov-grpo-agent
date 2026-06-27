# 基于 GRPO 的政务办理 Agent MVP

这是一个从零搭建的政务办理 Agent 训练闭环 MVP。当前版本不接真实政务接口，也不直接启动大模型训练；它先把数据契约、mock 政务环境、Agent Runtime、trajectory rollout、Verifier/Judge reward、GRPO 分组数据和评估指标跑通。

## 已实现内容

- 5 个 MVP 政务事项：租房提取公积金、医保异地备案、失业保险申领、人才补贴申请、个体工商户注册。
- 200 条 case 数据工厂，覆盖简单通过、信息缺失、资格不满足、材料缺失、复杂混合五类路径。
- 结构化动作空间：`Ask_User`、`Policy_Search`、`Eligibility_Check`、`Material_Check`、`Risk_Check`、`Submit`、`Refuse`。
- mock 工具环境：`Policy_Search`、`Eligibility_Check`、`Material_Check`。
- Agent Runtime：状态推进、动作校验、工具调用、trajectory logger、`max_turns=8`。
- SFT ChatML 样本生成：MVP 默认 2000 条。
- Reward 系统：规则 Verifier、轻量 Judge、missing-tool penalty、premature-submit penalty。
- GRPO 数据准备：按 case 分组，计算 reward mean/std 和组内相对 advantage。
- 评估指标：Success@1、Required Tool Recall、Premature Submit Rate、Missing Tool Rate、Material Check Call Rate、Final Decision Accuracy、Invalid Action Rate。

## 快速运行

```powershell
python -m gov_grpo_agent.cli --output-dir artifacts/mvp --case-count 200 --rollout-group-size 4
```

输出文件：

- `artifacts/mvp/cases.jsonl`
- `artifacts/mvp/sft_samples.jsonl`
- `artifacts/mvp/trajectories.jsonl`
- `artifacts/mvp/reward_reports.jsonl`
- `artifacts/mvp/grpo_groups.json`
- `artifacts/mvp/metrics.json`
- `artifacts/mvp/summary.json`

## 测试

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖数据分布、case schema、工具输出、Runtime 行为、reward penalty、GRPO advantage、SFT 样本和评估指标。

## 打包上传服务器

生成只包含源码、测试、文档和项目元数据的训练服务器上传包：

```powershell
python -m gov_grpo_agent.packaging --output dist/gov_grpo_agent_server_bundle.zip
```

压缩包会包含 `gov_grpo_agent/`、`tests/`、`docs/`、`README.md` 和 `pyproject.toml`，并排除 `.git/`、`artifacts/`、`__pycache__/`、`.venv/`、`dist/` 等本地产物。

上传到服务器后，可在服务器上解压并运行：

```bash
python -m unittest discover -s tests -v
python -m gov_grpo_agent.cli --output-dir artifacts/mvp --case-count 200 --rollout-group-size 4
```

## 下一阶段接入点

- 将 `sft_samples.jsonl` 转为 LLaMA-Factory、TRL 或 Axolotl 所需格式。
- 将 `grpo_groups.json` 接入 TRL GRPOTrainer、verl 或 OpenRLHF。
- 用 vLLM/SGLang 替换 `RuleBasedPolicy`，批量采样真实模型 rollout。
- 将 mock Policy DB 扩展为结构化政策库，后续再加入 BM25/BGE/FAISS 检索。

## Qwen3-8B SFT 起步命令

服务器建议先创建 Python 3.11 环境并安装训练依赖：

```bash
conda create -n govagent python=3.11 -y
conda activate govagent

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft trl bitsandbytes sentencepiece protobuf einops
```

先生成 MVP SFT 数据：

```bash
python -m gov_grpo_agent.cli --output-dir artifacts/mvp --case-count 200 --rollout-group-size 4
```

再用 Qwen3-8B 做第一轮 QLoRA SFT。你的服务器有 4 张 48GB 4090，建议先只使用 GPU 4-7：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m gov_grpo_agent.train_sft \
  --model-name-or-path Qwen/Qwen3-8B \
  --train-file artifacts/mvp/sft_samples.jsonl \
  --output-dir artifacts/qwen3_8b_sft_lora \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --num-train-epochs 1 \
  --lora-rank 16
```

如果你的环境能访问或已经下载了 `Qwen/Qwen3-8B-Instruct`，可把 `--model-name-or-path` 改成该模型 ID 或本地模型目录。第一轮目标是让模型学会合法 JSON 动作、工具调用顺序和追问逻辑，不追求最终效果最大化。

训练后验证 LoRA adapter 是否能输出合法动作 JSON：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python -m gov_grpo_agent.infer_sft \
  --model-name-or-path Qwen/Qwen3-8B \
  --adapter-path artifacts/qwen3_8b_sft_lora \
  --query "我想提取公积金交房租，应该怎么办？"
```

单条推理通过后，生成真实模型 rollout、reward、GRPO group 和指标：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=4 python -m gov_grpo_agent.model_rollout \
  --model-name-or-path Qwen/Qwen3-8B \
  --adapter-path artifacts/qwen3_8b_sft_lora \
  --output-dir artifacts/model_rollout \
  --case-count 200 \
  --rollout-group-size 4 \
  --max-turns 8 \
  --do-sample \
  --temperature 1.0 \
  --top-p 0.9
```

输出文件包括：

- `artifacts/model_rollout/model_trajectories.jsonl`
- `artifacts/model_rollout/model_reward_reports.jsonl`
- `artifacts/model_rollout/model_grpo_groups.json`
- `artifacts/model_rollout/model_metrics.json`
- `artifacts/model_rollout/model_summary.json`

`model_rollout` 会输出类似 `[rollout] 12/800 case=...` 的进度日志，并且每条 trajectory 完成后立即追加到 JSONL。可另开终端观察：

```bash
tail -f artifacts/model_rollout/model_trajectories.jsonl
```

也可以按 case 分片多卡并行。例如 8 张卡各跑 25 个 case：

```bash
python -m gov_grpo_agent.parallel_rollout \
  --gpus 0,1,2,3,4,5,6,7 \
  --total-cases 200 \
  --model-name-or-path Qwen/Qwen3-8B \
  --adapter-path artifacts/qwen3_8b_sft_lora \
  --output-root artifacts/parallel_model_rollout \
  --rollout-group-size 4 \
  --max-turns 8 \
  --do-sample \
  --temperature 1.0 \
  --top-p 0.9
```

先只打印将要启动的 worker 命令，不真正运行：

```bash
python -m gov_grpo_agent.parallel_rollout \
  --gpus 0,1,2,3,4,5,6,7 \
  --total-cases 200 \
  --dry-run
```

查看所有 worker 日志：

```bash
tail -f artifacts/parallel_model_rollout/gpu*/rollout.log
```

查看每个分片已完成的 trajectory 数：

```bash
watch -n 2 'for d in artifacts/parallel_model_rollout/gpu*; do echo -n "$d "; test -f "$d/model_trajectories.jsonl" && wc -l "$d/model_trajectories.jsonl" || echo 0; done'
```

所有分片完成后，合并 8 个 worker 的结果：

```bash
python -m gov_grpo_agent.merge_rollout_shards \
  --input-root artifacts/parallel_model_rollout \
  --output-dir artifacts/model_rollout_merged
```

合并后查看全局指标：

```bash
cat artifacts/model_rollout_merged/model_summary.json
cat artifacts/model_rollout_merged/model_metrics.json
```

准备框架无关的 GRPO 训练 JSONL 和质量报告：

```bash
python -m gov_grpo_agent.prepare_grpo \
  --input artifacts/model_rollout_merged_v3/model_grpo_groups.json \
  --output artifacts/grpo_train/qwen3_grpo_train.jsonl \
  --report artifacts/grpo_train/qwen3_grpo_report.json
```

报告字段包括：

- `groups`：case group 数量
- `responses`：总 rollout 数量
- `usable_groups`：reward 有方差、可提供组内相对优势信号的 group 数
- `low_variance_groups`：组内 reward 全相同或近似相同的 group 数
- `avg_reward`：平均 reward

如果 `usable_groups = 0`，说明同一个 case 的多条 rollout 没有 reward 差异，GRPO 没有相对优势信号。优先提高采样多样性，例如使用 `--do-sample --temperature 1.0 --top-p 0.9` 重新生成 rollout；不要把低方差 group 强行当作有效 GRPO 数据。

## verl GRPO 训练与可视化

准备 sampled GRPO JSONL 后，生成 verl 训练 job：

```bash
python -m gov_grpo_agent.train_grpo_verl \
  --input-jsonl artifacts/grpo_train/qwen3_grpo_train_sampled.jsonl \
  --work-dir artifacts/verl_grpo_qwen3_8b \
  --model-path Qwen/Qwen3-8B \
  --n-rollout 4 \
  --total-epochs 1 \
  --gpus 4,5,6,7
```

`--gpus` limits the generated verl job to the selected physical GPUs and sets
the worker count automatically. This is required on mixed-memory servers where
only a subset of devices has enough memory for GRPO training.

运行 verl：

```bash
bash artifacts/verl_grpo_qwen3_8b/run_verl_grpo.sh
```

生成本地 HTML/CSV 指标报告：

```bash
python -m gov_grpo_agent.metrics_report \
  --metrics artifacts/model_rollout_sampled_merged/model_metrics.json artifacts/verl_grpo_qwen3_8b/data/data_report.json \
  --grpo-report artifacts/grpo_train/qwen3_grpo_report_sampled.json \
  --output-dir artifacts/reports/qwen3_grpo \
  --title "Qwen3-8B GRPO Metrics"
```

导出 TensorBoard event：

```bash
python -m gov_grpo_agent.tensorboard_export \
  --metrics artifacts/model_rollout_sampled_merged/model_metrics.json artifacts/verl_grpo_qwen3_8b/data/data_report.json \
  --log-dir artifacts/tensorboard/qwen3_grpo

tensorboard --logdir artifacts/tensorboard/qwen3_grpo --host 0.0.0.0 --port 6006
```

详细说明见 `docs/verl_grpo_training.md`。

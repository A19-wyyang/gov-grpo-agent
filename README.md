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

# Government-Service Agentic RL with veRL

面向政务办理流程的可复现 Agentic RL 项目。模型需要通过多轮工具调用完成信息追问、政策查询、资格核验、材料检查和风险检查，最后才能提交或拒绝。正式训练使用 Qwen3-8B、LoRA 和 veRL GRPO；rollout 数可配置，双 RTX 4090 实跑配置为每个 case 4 条。

## 已实现

- 12 个政务事项、1,200 个结构化 case，按事项隔离为 800/200/200。
- 正常办理、信息缺失、资格不满足、材料缺失、风险和对抗指令六类场景。
- `PolicyView` 数据隔离，模型无法访问 `hidden_truth` 或 `expected_result`。
- 有状态多轮环境与统一 `government_service` 工具。
- veRL `BaseTool` adapter，每条 rollout 使用独立 episode。
- 硬事实、过程合规和表达质量分层奖励；错误提交和风险漏检硬门控。
- Assistant-only QLoRA SFT、veRL FSDP2 LoRA GRPO、可配置多 rollout。
- pass@1/pass@k、工具调用率、过早提交、风险提交和平均轮次评测。
- rollout/validation 自动导出 case、场景、环境奖励和安全指标。
- 11 个核心测试，覆盖数据切分、泄漏、参考流程、状态保持、环境奖励重放、Qwen Judge 和安全硬门控。

## 架构

```text
Official-guide catalog
        │
        ▼
1,200 CaseSpec ──► matter-isolated train/val/test
        │
        ├──► assistant-only SFT reference trajectories
        │
        └──► veRL ToolAgentLoop
                  │
          Qwen3-8B × N rollouts
                  │
          GovernmentServiceTool
                  │
       Verifier + Process + API Judge
                  │
          group-relative advantage
                  │
             LoRA update
```

## 本地验证

```powershell
$env:PYTHONPATH = "src"
python -m gov_agent_rl build-data --out data/processed --no-parquet
python -m gov_agent_rl validate-data --data data/processed
python -m pytest -q
```

生成 Parquet 需要 `datasets` 和 `pyarrow`：

```bash
pip install -e ".[data,dev]"
python -m gov_agent_rl build-data --out data/processed
```

## 服务器训练

目标目录：

```text
/data/code_repos/ywy/Project
```

目标环境：

```bash
conda activate govagent
export CUDA_VISIBLE_DEVICES=0,1
cd /data/code_repos/ywy/Project
pip install -e .
python -m gov_agent_rl build-data --out data/processed
python -m gov_agent_rl validate-data --data data/processed
```

服务器已验证的训练栈为 Python 3.11、PyTorch 2.7.0+cu118、
Transformers 4.56.1、PEFT 0.19.1、veRL 0.8.0 和 vLLM
0.9.1+cu118。`requirements-govagent.txt` 固定 Python 包版本；PyTorch 和
vLLM 必须安装与服务器 CUDA 11.8 匹配的官方 wheel，不能直接使用 PyPI
默认的 CUDA 12 wheel。

两个训练入口会先执行 `scripts/patch_verl_sdpa.py`。该脚本是幂等的，
用于给这一组已固定版本补齐 SDPA 转发、FSDP meta-device LoRA 加载、
PEFT 可选张量并行导入、vLLM API 兼容、状态化工具 request ID 以及
colocate 模式显存回收。Qwen3-8B 已缓存的
服务器默认使用离线模式；需要首次下载模型时可显式设置：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash scripts/train_sft.sh --smoke
```

### 1. SFT warm-up

```bash
bash scripts/train_sft.sh
```

训练只对 assistant 和 tool-call token 计算 loss。如果 Qwen chat template 不能返回 assistant mask，训练会直接失败，不会静默退化为全 token loss。

快速冒烟：

```bash
bash scripts/train_sft.sh --smoke
```

### 2. veRL GRPO

```bash
bash scripts/smoke_grpo.sh
bash scripts/train_grpo_formal.sh
```

双 RTX 4090 实跑参数：

- Base model：`Qwen/Qwen3-8B`
- LoRA：rank 16，alpha 32，all-linear
- Rollout：4 trajectories/case，10 个 GRPO steps
- Actor：FSDP2，2×RTX 4090
- Rollout engine：vLLM hybrid engine
- GRPO：clip 0.2、KL 0.02、entropy 0.001
- Horizon：最多 8 次工具动作
- Qwen3 thinking：关闭，避免推理文本挤占工具调用 token

正式 SFT adapter 从 `outputs/sft-qwen3-8b-formal/final_adapter` 加载。底层 `train_grpo.sh` 仍支持通过环境变量改为 8 rollouts 或更长训练，参见 `configs/experiment.env.example`。

`16 case × 4 rollout` 的长训练及训练后图表导出：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
TRAIN_BATCH_SIZE=16 \
PPO_MINI_BATCH_SIZE=16 \
ROLLOUT_N=4 \
TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=25 \
TEST_FREQ=25 \
VAL_BEFORE_TRAIN=True \
EXPERIMENT_NAME=qwen3_8b_gov_agent_grpo_b16r4_qwenjudge_100step \
bash scripts/run_grpo_with_plots.sh
```

训练结束后，`results/<experiment>/` 会保存 TensorBoard、rollout 和分场景指标
CSV，以及奖励安全、组内区分度、Judge Rubric、场景效果、优化稳定性和训练效率
PNG 曲线。验证集在训练前及每 25 steps 运行一次。

基于基线诊断进行单变量可归因优化时，可运行：

```bash
bash scripts/train_grpo_optimized.sh
```

该配置仍保持 `16 case × 4 rollout`，优先强化缺失必要工具时的 hard gate，
其中错误决策或危险提交归零，缺工具的最终答复上限为 0.1；同时惩罚非法工具参数、错误工具名和对不存在 slot 的追问；将 Judge 表达权重从
10% 降到 5%，并把 entropy coefficient 从 0.001 小幅提高到 0.002。它不会直接照搬更大的 group size；只有当 `group_reward_std` 和
`success@k` 证明有效轨迹覆盖不足时，才进一步增加 rollout 数。新增的
`exploration_coverage_metrics.png` 会记录 action 分布、success@k、安全/流程完整
success@k、缺工具最终答复率、错误工具名率和零方差 group 比例。对照实验使用相同
case 的配对 bootstrap 95% 置信区间；区间未完全跨过零的变化标记为
`inconclusive`，避免把采样噪声误报为提升。

### 3. 测试集评估

```bash
bash scripts/evaluate_grpo.sh
```

该命令从 `global_step_10` 恢复 GRPO checkpoint，在按事项隔离的 200 条
test case 上生成轨迹，并写出：

- `runs/qwen3_8b_gov_agent_grpo_test/validation/10.jsonl`
- `runs/qwen3_8b_gov_agent_grpo_test/metrics.json`

GitHub 仓库不提交 checkpoint 和完整运行目录；本次验收的聚合指标快照保存在
`results/qwen3_8b_grpo_formal_test_metrics.json`。

### 已验证实跑

- SFT smoke：真实非零学习率更新通过。
- GRPO smoke：1 step 完成，`actor/grad_norm=0.3301`、`actor/lr=5e-6`，4 条轨迹落盘。
- 正式 SFT：50 steps，`train_loss=1.0398`，`eval_loss=0.4040`，adapter 已保存。
- 正式 GRPO：10/10 steps 完成，共生成 160 条训练轨迹；末步
  `actor/grad_norm=0.1357`、`actor/lr=5e-6`，checkpoint 已保存到
  `checkpoints/gov_agent_rl/qwen3_8b_gov_agent_grpo_formal/global_step_10`。
- 独立测试集：从上述 checkpoint 恢复，在 200 个未见事项 case 上各生成
  1 条多轮轨迹，200/200 条均已落盘。

本次单种子工程验收结果：

> 以下数值来自升级 Qwen rubric Judge 之前的历史 checkpoint 与测试轨迹，保留用于可追溯对照。Judge/主奖励修复后的模型需要重新训练和评测，不能直接沿用这些数值。

| 指标 | 结果 |
| --- | ---: |
| pass@1 / 最终动作正确率 | 57.50% |
| 环境重放平均奖励 | 0.306125 |
| hard-gate 通过率 | 55.00% |
| 必要工具调用率 | 53.75% |
| 材料核验调用率 | 45.00% |
| 风险核验调用率 | 45.00% |
| 危险提交率 | 22.50% |
| 提前提交率 | 0.00% |
| 平均环境轮数 | 7.30 |

场景级 pass@1 为：对抗 100%（20 条）、不符合资格 50%（40 条）、
信息缺失 0%（40 条）、材料缺失 100%（30 条）、风险 100%（20 条）、
成功办理 50%（50 条）。这暴露出当前模型对信息缺失追问以及部分事项的
提交/拒绝决策仍有明显短板，不能把本次工程验收结果表述为生产可用。

评测每个 case 只有 1 条 rollout，因此本次 `pass@k` 与 `pass@1` 相同，
不把它冒充为 pass@8。完整原始轨迹和聚合指标分别保存在：

- `runs/qwen3_8b_gov_agent_grpo_test/validation/10.jsonl`
- `runs/qwen3_8b_gov_agent_grpo_test/metrics.json`

## 奖励约束

总奖励由硬事实 0.65、过程合规 0.25、表达质量 0.10 组成。

硬事实包括：

- 最终动作正确
- 必要槽位完整
- 已完成资格、材料和风险核验
- 工具结果与最终动作一致

错误提交、风险漏检或错误最终动作会触发 hard gate，总奖励上限为 0.2。Qwen Judge 只评价表达质量，不能推翻 Verifier。veRL 的主 `score` 使用完整环境重放奖励，不再使用独立的启发式文本分。

默认 Judge 使用阿里云百炼 `qwen3.7-max`，rubric 为：

| 维度 | 权重 | 边界 |
| --- | ---: | --- |
| 清晰度 | 20% | 表述清楚、简洁、无歧义 |
| 理由完整性 | 25% | 说明决定理由，不判断事实真伪 |
| 可执行性 | 25% | 给出明确下一步 |
| 决策一致性 | 20% | 文本与 `SUBMIT/REFUSE` 一致 |
| 专业性 | 10% | 专业、尊重、不夸大承诺 |

配置百炼 OpenAI-compatible Judge：

```bash
export GOV_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export GOV_JUDGE_API_KEY=你的百炼密钥
export GOV_JUDGE_MODEL=qwen3.7-max
export GOV_JUDGE_REQUIRED=1
```

也可以将这些变量写入服务器私有的 `.env.judge`；`train_grpo.sh` 会自动加载，该文件受 `.gitignore` 保护。Judge 返回五个维度的 0-4 分，最终归一化到 0-1，并缓存到 SQLite。未配置 API 时表达分记为缺失；正式训练建议设置 `GOV_JUDGE_REQUIRED=1`，避免接口异常时静默降级。

## 数据说明

`catalog.py` 中的公开来源是可追溯种子，不等于已经完成法律审核。正式对外发布指标前必须：

1. 重新抓取并归档办事指南正文；
2. 更新来源内容哈希和版本日期；
3. 人工核对资格、材料与风险规则；
4. 删除真实身份证号、联系方式等敏感信息；
5. 保证同事项及其派生变体不跨训练和测试集合。

测试集按事项隔离，衡量未见事项的迁移能力，因此不能用测试事项生成训练变体。

## 结果纪律

- 不在代码或 README 中预写“提升百分比”。
- 正式实验至少运行 3 个随机种子并报告均值与标准差。
- 必须同时报告 pass@1、pass@8、错误提交率、missing-tool 率和必要工具调用率。
- 当前 10-step、单种子、每 case 1 rollout 的结果只用于工程链路验收，不满足上一条论文级统计要求。
- 未达到预期也保留原始轨迹和消融结果。

## 代码入口

- `src/gov_agent_rl/schema.py`：严格 schema 与 PolicyView
- `src/gov_agent_rl/agent_env.py`：状态机与动作执行
- `src/gov_agent_rl/rewarding.py`：在线/离线统一奖励
- `src/gov_agent_rl/verl_tool.py`：veRL stateful tool
- `scripts/train_sft.py`：assistant-only QLoRA SFT
- `scripts/train_grpo.sh`：veRL Agent GRPO

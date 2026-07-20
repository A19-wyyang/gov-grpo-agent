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
- 60+ 项自动化测试，覆盖数据切分、泄漏、参考流程、状态保持、环境奖励重放、Qwen Judge、安全硬门控、checkpoint 完整性和 rollout 覆盖。

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

每次 GRPO 在加载 veRL 前都会创建
`runs/<experiment>/run_manifest.json`，其中包含代码 commit、数据 manifest 与文件
SHA-256、SFT adapter 目录指纹、工具配置、reward 定义、Judge 模型以及关键 GRPO
超参数，但绝不记录 API key。同名 experiment 再次启动时必须与该 manifest 完全一致；
若数据版本、adapter、horizon、rollout N、学习率或奖励配置发生漂移，会写出
`run_manifest.candidate.json` 并拒绝继续，避免把不同实验拼进同一条曲线。

### 1. SFT warm-up

```bash
bash scripts/train_sft.sh
```

训练只对 assistant 和 tool-call token 计算 loss。如果 Qwen chat template 不能返回 assistant mask，训练会直接失败，不会静默退化为全 token loss。
训练前还会生成 `sft_data_audit.json`，按场景记录总 token 和 assistant target token
的均值、P50、P95 与最大值。任何样本超过 `--max-length` 都会直接失败，不再静默截掉
轨迹尾部的核验或最终决策。训练完成后，`scenario_eval_metrics.json` 会分别保存六类
场景的 teacher-forced eval loss，用于判断 missing-information 短板是否已存在于
SFT warm-up，而不是盲目归因给 GRPO。

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
`tensorboard_metric_coverage.json` 会记录各逻辑指标实际匹配到的 veRL tag；兼容
`actor/entropy`/`actor/entropy_loss`、不同 timing 和显存前缀。优化图将
`actor/kl_loss`（相对 reference 的 KL 约束）与 `actor/ppo_kl`（一次策略更新幅度）
分开显示。训练成功但 policy loss、两类 KL、entropy、grad norm、clip fraction 或
学习率曲线缺失时，导出流程会失败，不再交付带空面板的“完整图表”。
同一流程还会读取最新 step 的真实同-case rollout，生成
`rollout_budget_analysis.json/csv/png`。它以当前每个 case 的成功比例保守投影
N=8/16 的 success@N 和 mixed-outcome group rate，并同时显示相对采样成本。零成功
case 不会因为统计平滑被虚构成“增加采样即可成功”；只有流程安全覆盖和有效组内对比
预计同时达到门槛时，才建议运行更大 N 的消融。

基于基线诊断进行第一轮 reward-v2 优化时，可运行：

```bash
bash scripts/train_grpo_optimized.sh
```

该配置仍保持 `16 case × 4 rollout`，优先强化缺失必要工具时的 hard gate，
其中错误决策或危险提交归零，缺工具的最终答复上限为 0.1；同时惩罚非法工具参数、错误工具名和对不存在 slot 的追问；将 Judge 表达权重从
10% 降到 5%，并把 entropy coefficient 从 0.001 小幅提高到 0.002。它不会直接照搬更大的 group size；只有当 `group_reward_std` 和
`success@k` 证明有效轨迹覆盖不足时，才进一步增加 rollout 数。新增的
`exploration_coverage_metrics.png` 会记录 action 分布、success@k、安全/流程完整
success@1 与 success@k、缺工具最终答复率、错误工具名率和零方差 group 比例。
这里的流程完整不再仅表示“必要工具曾经出现过”：只有政策查询、资格核验、材料检查、
风险检查按顺序执行，工具不重复调用，必要信息补齐后再做资格核验，且工具结果支持
正确的最终动作时，才记为 `process_compliant=1` 并进入 `process success@k`。
旧 rollout 若没有该字段仍可按必要工具覆盖率回退读取；使用统一 Verifier 重放后则
一律采用新口径。工具乱序、重复调用和过早资格核验分别记录独立曲线，并纳入 test
晋级门槛。
工具调用 JSON 截断、缺字段或参数无法解析时也会计入 `illegal_action`，并记录
`tool_call_format_error_rate`，不会从 reward 轨迹中静默消失。每次非法调用都会占用
horizon 并累计惩罚，同时记录 `illegal_action_count` 和
`illegal_action_attempt_rate`；若候选模型的非法调用、错误工具名或格式错误显著恶化，
测试集晋级门槛会直接拒绝该 checkpoint。
首次 `SUBMIT/REFUSE` 后继续生成的工具调用也不会再被 Verifier 静默忽略：
`trailing_action_count/rate` 会累计这些调用、降低过程奖励，并作为 checkpoint 晋级
门槛，从而区分“需要更多合理纠错空间”和“已经结束仍持续循环”。
对照实验使用相同 case 的配对 bootstrap 95% 置信区间；区间未完全跨过零的变化标记为
`inconclusive`，避免把采样噪声误报为提升。

Judge API 或格式校验失败时不会再给予中性表达分，默认回退为 0，并记录
`judge_fallback_used`；可通过 `GOV_JUDGE_ERROR_LOG` 保存不含请求正文和密钥的
异常类型，避免接口失败成为 reward shortcut。
Verifier 会先于 Judge 执行；任何 decision/process hard gate 都直接跳过远程 Judge，
且表达分对 hard-gated reward 的贡献强制为 0，即使诊断配置使用非零 gate ceiling
也不能用措辞挽救事实或流程错误。`judge_used`、`judge_fallback_used`、
`judge_skipped_hard_gate` 和 `judge_empty_message` 分开统计，空回复的本地零分不再
冒充 Qwen Judge 覆盖率。

由于 reward-v2 改变了奖励定义，A/B 报告不会直接比较两次训练各自保存的 reward。
`rescore_rollouts.py` 会先在同一 reward-v2 Verifier 下离线重放两边轨迹，保留原始
分数为 `source_environment_reward`，再计算配对差值与置信区间。
候选实验每 5 步验证并保存一次；`select_best_grpo_checkpoint.py` 先最大化六类场景中
最弱场景的 `process pass@1` Wilson 95% 下界，再依次考虑总体 process/safe
`pass@1` 下界、危险提交率、非法动作率，最后才使用 `pass@k` 探索覆盖、最终动作
正确率和统一 reward，并生成
`checkpoint_selection.csv/json/png`。这样不会为了总体均值牺牲信息缺失等弱场景。
validation 仅用于选择 checkpoint；随后 baseline 与候选最佳 checkpoint 都会在按事项
隔离的 test 集上运行。`decide_grpo_promotion.py` 只有在 test 上流程安全成功率或安全
`pass@1` 显著改善，且 `pass@k`、unsafe、hard gate、最终动作等指标无显著退化时才
输出 `promote`；
否则输出 `reject` 或 `needs_more_evidence`。晋级门槛同时检查各场景的配对置信区间，
并生成 `scenario_comparison.csv/png`，避免总体均值掩盖风险、缺材料或对抗场景退化。
自动切换前，`check_grpo_snapshot.py` 还会验证 validation 恰好包含 200 个唯一 case、
step 一致，并确认两份模型、优化器、extra-state 分片以及 tokenizer/config 均非空；
全部文件经过 30 秒稳定期后才允许切换，不再仅凭目录存在或 JSONL 非空就停止
baseline。

历史独立测试中 `missing_information` pass@1 为 0%，平均环境轮数达到 7.3；同时
140 个参考 case 的标准流程恰好需要 8 个动作，已顶满默认 horizon=8。为验证是否由
完成空间不足造成，可在 reward-v2 结束后运行受控消融：

```bash
bash scripts/train_grpo_horizon10.sh
```

该实验保持 B16×R4、reward-v2、entropy、学习率和训练步数不变，只将工具环境
horizon 从 8 调至 10。`horizon_metrics.png` 会按场景记录平均动作尝试数和
`max_steps_exceeded`。只有 missing-information 的流程安全成功率改善，且循环、
非法调用和安全指标不退化时，才保留这一修改。该候选先使用 validation 选择，不重复
查看 test；最终只对选定配置运行一次独立 test。

### 3. 测试集评估

```bash
bash scripts/evaluate_grpo.sh
```

该命令从 `global_step_10` 恢复 GRPO checkpoint，在按事项隔离的 200 条
test case 上默认各生成 4 条轨迹，并写出：

- `runs/qwen3_8b_gov_agent_grpo_test/validation/10.jsonl`
- `runs/qwen3_8b_gov_agent_grpo_test/metrics.json`

可通过 `EVAL_ROLLOUT_N` 修改测试采样数。指标聚合同时报告 pass@1、pass@k、
safe pass@k 和流程完整 pass@k；进入 A/B 报告前必须通过覆盖校验，即 200 个 case
均存在且每个 case 恰好具有相同的 k 条 rollout。checkpoint 选择中的 Wilson 区间
也按 case 数而非 rollout 数计算，避免把同一 case 的多次采样误当成独立样本。
N>1 的 validation/test 明确使用 `do_sample=True`、temperature 0.7、top-p 0.95，
并以固定 `EVAL_SEED` 复现；不再把确定性贪心输出复制 N 次后称为 pass@N。
`unique_output_rate` 和 `identical_output_group_rate` 会同时进入 CSV、A/B 报告及
`validation_metrics.png`，用于识别采样退化或同 case 全部输出相同的情况。

GitHub 仓库不提交 checkpoint 和完整运行目录；本次验收的聚合指标快照保存在
`results/history/qwen3_8b_grpo_formal_test_metrics.json`。该文件属于旧 reward
口径，只用于历史复现，不作为当前候选晋级依据。

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

历史基线总奖励由硬事实 0.65、过程合规 0.25、表达质量 0.10 组成。
当前 reward-v2 分别调整为 0.70、0.25、0.05，以降低表达分对硬事实和流程信号的
稀释；离线 A/B 会用同一 reward-v2 Verifier 重放两边轨迹。

硬事实包括：

- 最终动作正确
- 必要槽位完整
- 已完成资格、材料和风险核验
- 工具结果与最终动作一致

错误提交、风险漏检或错误最终动作会触发 hard gate，总奖励上限为 0.2。Qwen Judge 只评价表达质量，不能推翻 Verifier。veRL 的主 `score` 使用完整环境重放奖励，不再使用独立的启发式文本分。
`unsafe_submit` 由 Verifier 的权威最终决策判定，因此即使模型跳过资格或材料工具，
“资格不符/材料缺失却直接提交”也不会从安全指标中漏报。
`tool_results_support_final` 会进一步验证：只有资格、材料和风险全部通过时才支持
`SUBMIT`，任一不通过才支持 `REFUSE`。模型即使命中数据标签，只要最终动作与实际
工具结果冲突，也会触发 `tool_result_conflict` hard gate；该指标进入训练曲线、场景
对比和 checkpoint 晋级门槛。

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

仓库同时保留两个严格隔离的数据版本：

- `data/processed`：`legacy_v1`，精确重建历史 baseline/reward-v2 使用的低多样性
  case，仅用于完成已有实验和可追溯重放。其训练 split 已审计出 20 条
  `provident_fund_loan/ineligible` 标签与工具结果冲突，因此历史指标必须披露这一限制。
- `data/processed_v2`：`diverse_v2`，保持 12 个事项、六类场景和 800/200/200
  按事项隔离切分，但通过请求表达、可见槽位组合及合规数值变化增加有效样本量，并
  修复历史 ineligible case 中可见值与隐藏真值不一致的问题。

v2 训练集从 24 种可见输入提升到 526 种，800 个完整 case 全部唯一；
validation/test 分别有 128/136 种可见输入且均为 200/200 完整唯一。每份 manifest
包含 `dataset_variant` 和整体 case SHA-256。新 rollout 还记录逐 case 指纹，离线
Verifier 遇到同 ID 不同内容会直接失败。历史无指纹轨迹只有在使用精确归档的
`legacy_v1` 时才能显式启用 `--allow-missing-case-fingerprint`。
`diverse_v2` 的 train/validation/test 共 1,200 条均通过工具结果一致性审计，冲突为 0。

重建命令：

```bash
python -m gov_agent_rl build-data --out data/processed --legacy-case-variants
python -m gov_agent_rl build-data --out data/processed_v2
python scripts/audit_case_diversity.py \
  --cases data/processed_v2/train.cases.jsonl \
  --require-full-unique
```

完成 legacy 实验后，v2 数据消融按以下顺序运行：

```bash
bash scripts/train_sft_diverse_v2.sh
bash scripts/train_grpo_diverse_v2.sh
bash scripts/compare_diverse_v2_on_fixed_test.sh
```

v2 GRPO 保持 reward-v2、H8、B16×R4 和优化器设置不变。最后一个脚本先在
`diverse_v2` validation 上选择 checkpoint，再让 legacy 最佳模型与 v2 最佳模型同时
在同一份 `diverse_v2` test 上各生成 4 条 rollout；两边使用相同 Verifier 且强制
case 指纹匹配。这样最终差异主要归因于训练数据管线，而不是两套测试集难度不同。

若 diverse-v2 的 validation 显示某些场景（尤其信息缺失）流程成功率明显落后，可运行：

```bash
bash scripts/train_grpo_curriculum_v3.sh
bash scripts/compare_curriculum_v3_on_fixed_test.sh
```

该消融不会读取 test，也不会盲目增加每个 case 的 rollout。脚本先用当前 reward-v2
Verifier 重放 baseline validation，再优先读取场景级 `process_pass_at_1`，按成功缺口
生成有上限的训练重复权重；默认单场景最多
2 倍、总训练集最多扩张 1.5 倍，并保持原始训练样本全部存在。生成的
`data/curriculum_v3/manifest.json` 会记录 validation 指标来源、输入文件哈希、
各场景分数、倍率和实际追加数量。训练仍保持 reward-v2、H8、B16×R4、25 steps，
因此可与 diverse-v2 做单变量配对比较；只有固定 validation/test 上的流程成功率
改善且安全、工具协议指标不退化时才保留。
课程权重使用 `@1` 而不是 `@k`，因为后者会随采样数增加而自然上升，适合衡量探索
覆盖，却可能掩盖单条策略成功率仍然偏低。

SFT 侧另有一项独立的 loss 消融：

```bash
bash scripts/train_sft_turn_balanced.sh
bash scripts/train_grpo_turn_balanced_sft.sh
bash scripts/compare_turn_balanced_sft_on_fixed_test.sh
```

普通 causal-LM SFT 按 token 求平均，较长的 query、slot 或最终 message 会自然占据
更多 loss 权重；但政务 Agent 的每次工具决策同样重要。`--turn-balanced-loss` 会先在
每个 assistant 决策轮次内部平均 token CE，再跨所有决策轮次平均，使短的资格、
材料、风险动作不会被长文本稀释。`sft_data_audit.json` 同时记录场景级 assistant
轮次数和轮次 token 长度不平衡度。该实验只替换 SFT adapter，后续 GRPO 继续使用
diverse-v2、reward-v2、H8、B16×R4 和 25 steps；固定 v2 test 比较通过后才能与
GRPO curriculum 组合。

正式 SFT 不再无条件使用最后一步。默认每 25 optimizer steps 在按事项隔离的
validation 上评估并保存，训练结束恢复最低 `eval_loss` 的 checkpoint 后才写入
`final_adapter`；如果正式训练没有产生任何可评估 checkpoint，流程直接失败。
输出目录额外保存 `training_log.json`、`sft_training_summary.json`、
`sft_training_metrics.png` 和 `sft_scenario_metrics.png`，分别用于检查 train/eval
loss、学习率、最佳 checkpoint 及各场景验证损失。

GRPO policy loss 另提供长度归一化消融：

```bash
bash scripts/train_grpo_sequence_balanced.sh
bash scripts/compare_sequence_balanced_on_fixed_test.sh
```

baseline 显式使用 veRL 默认 `token-mean`，候选仅改为
`seq-mean-token-mean`，使每条 rollout 先做内部 token 平均，再跨 rollout 平均。
该模式接近原始 GRPO 的 sample-level loss，但不会直接替换默认值，因为长回复下可能
增加方差。`length_bias_metrics.png` 会同时展示训练/验证输出字符长度、长度与 reward
的 Pearson 相关系数，以及最长和最短四分位的 reward 差；只有弱场景提升且 KL、
entropy、clip fraction、安全指标无退化时才保留。实际 loss aggregation 也写入
run manifest，避免两次实验口径不明。

针对 GRPO 组内 reward 完全相同、归一化 advantage 为零的问题，另提供
informative-group 动态采样消融：

```bash
bash scripts/train_grpo_informative_groups.sh
bash scripts/compare_informative_groups_on_fixed_test.sh
```

baseline 默认关闭 group filtering；候选使用 veRL/DAPO 的 group filtering，按
`score` 丢弃零方差组，并继续生成候选 case，直到凑足 16 个有效组或达到 3 个
generation batches。其余条件仍为 diverse-v2、reward-v2、同一 SFT adapter、H8、
B16×R4、25 steps 和 `token-mean`。候选的 `gen_batch_size=24`，因此每个 actor
update 的配置生成上限为 `(24 / 16) × 3 = 4.5×`；这是最坏情况上限，不代表真实
开销。`group_learning_metrics.png` 分别报告 informative/zero-variance group rate、
rollout dump 中可观察到的 group multiplier、配置 batch ratio 和配置上限，正式比较
还必须报告 wall time。只有 informative group rate 提升，并且固定 v2 test 的 reward、
process/safe success 和弱场景指标改善，同时额外开销可接受时才保留该方案。

另提供与动态采样完全分离的 DAPO Clip-Higher 消融：

```bash
bash scripts/train_grpo_clip_higher.sh
bash scripts/compare_clip_higher_on_fixed_test.sh
```

baseline 的上下 clip ratio 都是 `0.2`；候选只把正向更新上界放宽到 `0.28`，下界仍为
`0.2`。这样可以检验高 reward 轨迹是否被对称 clipping 过早截断，同时避免把收益错误
归因于数据或 reward 变化。`optimization_metrics.png` 会分别记录 overall、lower 和
higher clip fraction，并联动检查 policy-update KL、reference KL、grad norm 与
entropy。只有固定 v2 test 和弱场景指标改善，且 KL、错误提交率和训练稳定性不退化时
才保留；不会先与 informative-group 或 curriculum 组合。

普通候选通过单 seed 固定 test 筛选后，使用统一的多 seed 运行器：

```bash
SCREENING_DECISION_FILE=results/comparisons/<single-seed>/fixed_v2_test/promotion_decision.json \
  SEEDS="42 43 44" bash scripts/run_clip_higher_multiseed.sh

SCREENING_DECISION_FILE=results/comparisons/<single-seed>/fixed_v2_test/promotion_decision.json \
  SEEDS="42 43 44" bash scripts/run_informative_groups_multiseed.sh
```

运行器拒绝单 seed 已判定为 `reject/invalid` 的候选，强制至少三个 seed，并让每对
baseline/candidate 使用相同训练与 rollout seed。最终晋级要求 `process/safe
pass@1` 跨 seed 改善，且 `pass@k`、安全、工具协议和各场景指标无已解析退化。

正式结论使用三个独立 GRPO seed：

```bash
SEEDS="42 43 44" bash scripts/run_curriculum_multiseed.sh
```

`TRAIN_SEED` 会同时控制 veRL 的训练集 RandomSampler、actor/ref FSDP 初始化、
actor mini-batch loader 和 vLLM rollout，并写入 immutable run manifest。每个 seed
内部先比较相同 seed 的 diverse-v2 baseline 与 curriculum 候选，再由
`aggregate_seed_comparisons.py` 汇总 seed 级差值、标准差、95% t 区间和方向一致率，
输出 `multiseed_comparison.json`、`multiseed_metrics.csv` 与
`multiseed_comparison.png`。只有主要指标跨 seed 显著改善且没有总体或场景安全退化，
才允许输出 `promote`。

## 结果纪律

- 不在代码或 README 中预写“提升百分比”。
- 正式实验至少运行 3 个随机种子并报告均值与标准差。
- 必须同时报告 pass@1、当前采样数对应的 pass@k、错误提交率、missing-tool 率和必要工具调用率。
- 当前 10-step、单种子、每 case 1 rollout 的结果只用于工程链路验收，不满足上一条论文级统计要求。
- 未达到预期也保留原始轨迹和消融结果。

## 代码入口

- `src/gov_agent_rl/schema.py`：严格 schema 与 PolicyView
- `src/gov_agent_rl/agent_env.py`：状态机与动作执行
- `src/gov_agent_rl/rewarding.py`：在线/离线统一奖励
- `src/gov_agent_rl/verl_tool.py`：veRL stateful tool
- `scripts/train_sft.py`：assistant-only QLoRA SFT
- `scripts/train_grpo.sh`：veRL Agent GRPO

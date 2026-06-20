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

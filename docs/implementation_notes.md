# 政务办理 Agent MVP 实施说明

## 当前边界

本仓库实现的是工程闭环 MVP：数据、环境、Runtime、rollout、reward、GRPO 分组和评估。真实模型训练、显存调度、vLLM 服务和 LoRA checkpoint 产出属于下一阶段集成。

## 模块结构

- `gov_grpo_agent.data`：5 个事项、200 条 MVP case、结构化政策目录。
- `gov_grpo_agent.schemas`：case、action、trajectory 的最小契约校验。
- `gov_grpo_agent.tools`：mock 政策查询、资格核验、材料检查。
- `gov_grpo_agent.runtime`：规则基线 Agent 和多轮 trajectory 运行器。
- `gov_grpo_agent.sft`：ChatML SFT 样本构造。
- `gov_grpo_agent.rewards`：Verifier/Judge/penalty reward 聚合。
- `gov_grpo_agent.grpo`：按 case 分组并计算组内 advantage。
- `gov_grpo_agent.evaluation`：核心评估指标。
- `gov_grpo_agent.cli`：一键生成 MVP artifacts。

## MVP 验收口径

1. `python -m unittest discover -s tests -v` 全部通过。
2. CLI 生成 200 条 case、2000 条 SFT 样本、800 条 trajectory、800 条 reward report、200 个 GRPO group。
3. `metrics.json` 中 `required_tool_recall` 应为 `1.0`，`missing_tool_rate` 应为 `0.0`。
4. `reward_reports.jsonl` 中完整工具链 trajectory 的 reward 应显著高于缺工具或过早提交的轨迹。

## 正式版扩展策略

1. 扩展事项到 20 个，并保持 path_type 分层采样。
2. 将 SFT 样本扩展到约 10000 条，保持动作格式和多轮工具调用样本占主导。
3. 将 `RuleBasedPolicy` 替换为真实模型采样器，保留 Runtime、tool registry 和 trajectory schema。
4. 将 `grpo_groups.json` 转为训练框架输入，使用组内 reward 标准化和 KL 约束进行离线迭代式 GRPO。

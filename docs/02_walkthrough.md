# 代码走读顺序

## 1. 从 case 开始

先打开 `data/cases/social_subsidy_refuse.json`。这个 case 的资格满足，但缺少 `employment_certificate`，因此标准结果是 `REFUSE`。

## 2. 看环境如何推进

打开 `src/gov_agent_rl/environment.py`：

1. `ASK_USER` 从 `hidden_truth` 返回槽位。
2. 工具调用结果写入 `state.tool_results`。
3. `SUBMIT` 和 `REFUSE` 终止 trajectory。
4. 环境允许错误动作发生，只记录 failure tag。这样离线训练才能看到负样本。

## 3. 对比两条 trajectory

打开 `runs/demo/trajectories.jsonl`，搜索：

- `subsidy_001_run_21_careful_policy`
- `subsidy_001_run_25_judge_hacker_policy`

`careful_policy` 调用了材料检查，发现证明缺失后拒答。`judge_hacker_policy` 给出流畅建议，但跳过材料和风险检查。

## 4. 看奖励如何识别漏洞

打开 `runs/demo/experiments/reward_hacking_comparison.jsonl`。同一条 hacking trajectory 会分别经过：

- `collapse_prone`：Judge 权重偏高，且未强制必要工具。
- `hardened`：Verifier 检测 `MATERIAL_CHECK` 和 `RISK_CHECK` 缺失并处罚。

## 5. 看组内 advantage

打开 `runs/demo/grpo_groups.jsonl`。每个 case 内部单独计算平均 reward 和标准差。好的轨迹 advantage 为正，差的轨迹 advantage 为负。

## 6. 看坍塌监控

打开 `runs/demo/experiments/report.md` 和两个 history JSONL：

- `before_fix.jsonl`
- `after_fix.jsonl`

重点观察四个指标：

| 指标 | 含义 |
| --- | --- |
| `policy_entropy` | 动作或策略多样性 |
| `mean_group_reward_std` | 组内对比信号强度 |
| `required_tool_rate` | 必要工具调用率 |
| `final_action_distribution` | 是否集中到 `REFUSE` 或过早 `SUBMIT` |

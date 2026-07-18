# 面试讲解手册

## 1. 90 秒项目介绍

我做的是政务办理 Agent 的策略优化。和普通问答不同，办理流程需要多轮决策：先识别诉求，再补齐槽位，查询政策，做资格、材料和风险核验，最后提交或拒答。

我先把政务事项结构化为 case schema，并定义 `ASK_USER`、`POLICY_SEARCH`、`ELIGIBILITY_CHECK`、`MATERIAL_CHECK`、`RISK_CHECK`、`SUBMIT`、`REFUSE` 动作空间。Agent 和环境交互后，每一步都会记录 `state/action/observation`，形成 trajectory。

奖励侧使用 Verifier 和 Judge 两层信号：Verifier 检查硬事实和必要工具，Judge 只评价最终回复的清晰度和可执行性。训练时对同一 case 做多条 rollout，用组内相对 reward 计算 GRPO advantage。

工程链路已经使用 Qwen3-8B 完成 assistant-only QLoRA SFT、veRL GRPO 和 200 条独立测试集评测。奖励函数通过环境重放检查必要工具、最终动作和危险提交，表达 Judge 不能覆盖这些硬事实。

## 2. 高频追问

### 为什么用 GRPO？

因为一个 case 可以天然采样多条办理路径。GRPO 用同组 rollout 的相对 reward 构造 advantage，不需要单独训练 value model，适合比较“问得完整但路径稍长”和“回复流畅但跳步骤”等策略。

### Verifier 和 Judge 为什么分开？

硬事实不能交给 Judge。槽位是否补齐、材料检查是否调用、风险是否核验都应该规则化。Judge 只补充表达质量，否则模型容易利用语言偏好做 reward hacking。

### 为什么环境允许错误提交？

训练环境必须让错误动作真实发生并留下标签。如果在环境层直接禁止错误提交，策略学不到错误行为为什么低分，也无法回放线上失败模式。

### 怎么发现策略坍塌？

同时看动作分布、`REFUSE/SUBMIT` 占比、必要工具调用率、policy entropy 和组内 reward 方差。只有平均 reward 不够，因为错误奖励设计可能让坍塌策略拿到高分。

### entropy bonus 起什么作用？

它鼓励保留策略多样性，避免过早集中到单一路径。它不能替代 reward 修复：如果奖励方向错了，只加 entropy 仍然会围绕错误目标探索。

### 当前实验有什么边界？

当前已经完成真实 token rollout、clipped objective、KL 约束和双 RTX 4090 反向传播，但正式 GRPO 只有 10 steps，测试为单种子且每个 case 只有一条 rollout。因此它证明了完整工程链路可运行，不代表模型充分收敛，也不满足论文级多种子统计要求。

## 3. 不要说过头

- 可以陈述已经实跑 Qwen3-8B SFT 和 veRL GRPO，但必须同时说明 10-step、单种子和测试 rollout 数量等实验边界。
- 不要把 Judge 说成事实裁判。
- 不要只说“加大惩罚”，要说清具体监控指标和因果链。
- 不要把 GRPO 描述成只做 reward 排序；核心是同组标准化 advantage 驱动策略更新。

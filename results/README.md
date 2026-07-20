# Results layout

- `audits/`：数据多样性与规则一致性审计，可由 `audit_case_*.py` 重建。
- `history/`：旧模型、旧 reward 口径的只读结果快照，不参与当前晋级。
- `comparisons/`：固定测试集 A/B 和多 seed 汇总，运行实验后生成。
- `<experiment>/`：TensorBoard 导出、rollout 指标和可视化，运行实验后生成。

模型权重、checkpoint、原始运行日志和密钥不放在此目录，也不提交到 GitHub。

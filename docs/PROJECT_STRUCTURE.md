# 项目目录说明

## 应提交的工程文件

- `src/gov_agent_rl/`：Agent 环境、schema、数据构建、Verifier reward、Qwen Judge、veRL 适配和评估。
- `scripts/`：SFT/GRPO 主入口、受控消融、固定测试比较、指标导出和运行完整性检查。
- `configs/`：工具 horizon 和无密钥实验配置示例。
- `tests/`：环境、reward、数据、SFT、GRPO 指标与晋级规则测试。
- `data/processed/`：legacy-v1，用于复现基线和 diverse-v2 数据消融。
- `data/processed_v2/`：当前推荐的 diverse-v2 固定 train/validation/test。
- `results/audits/`：数据审计证据。
- `results/history/`：不参与当前结论的历史结果。

## 本地生成且不提交的文件

- `runs/`：rollout、validation 和训练日志。
- `outputs/`：SFT adapter 等模型产物。
- `checkpoints/`：veRL checkpoint。
- `tmp/`、`qa_render/`：临时数据和渲染检查。
- `.pytest_cache/`、`__pycache__/`：测试和 Python 缓存。
- `.env*`：Judge API 等本机密钥配置。

## 入口关系

1. `python -m gov_agent_rl build-data` 构建数据。
2. `scripts/train_sft.py` 训练 SFT adapter。
3. `scripts/train_grpo.sh` 是唯一底层 GRPO 入口。
4. `train_grpo_*.sh` 只覆盖单个受控变量。
5. `compare_*_on_fixed_test.sh` 使用 validation 选点，再在固定 v2 test 比较。
6. `aggregate_seed_comparisons.py` 汇总至少三个配对 seed 后决定是否晋级。

不要把旧 reward 指标、单 seed 波动或 `pass@k` 的自然采样增益写成最终提升。

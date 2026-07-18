from __future__ import annotations

from pathlib import Path

from .io_utils import load_cases, read_jsonl


def generate_report(
    cases_dir: Path,
    trajectories_path: Path,
    scores_path: Path,
    groups_path: Path,
    output_path: Path,
) -> Path:
    cases = {case.case_id: case for case in load_cases(cases_dir)}
    trajectories = read_jsonl(trajectories_path)
    scores = {row["trajectory_id"]: row for row in read_jsonl(scores_path)}
    groups = read_jsonl(groups_path)
    trajectories_by_case = {}
    for trajectory in trajectories:
        trajectories_by_case.setdefault(trajectory["case_id"], []).append(trajectory)

    lines = [
        "# 政务办理 Agentic RL 流程报告",
        "",
        "本报告由本地 demo 自动生成，用来展示 case 样本如何流经 rollout、verifier/judge、reward 和 GRPO group。",
        "",
        "## 数据流",
        "",
        "1. `data/cases/*.json` 提供 Agent 可见信息、环境隐藏真值、政策规则和标准结果。",
        "2. `trajectories.jsonl` 记录每个策略在同一 case 上的多轮决策轨迹。",
        "3. `scored.jsonl` 保存 verifier 硬事实分、judge 表达分、惩罚标签和 reward。",
        "4. `grpo_groups.jsonl` 按 case 聚合同组轨迹，形成组内高低分对比样本。",
        "5. `experiments/report.md` 对照复现策略坍塌、entropy bonus 修复和 reward hacking。",
        "",
        "## Case 结果",
        "",
    ]

    for group in groups:
        case = cases[group["case_id"]]
        lines.extend(
            [
                f"### {case.case_id} - {case.title}",
                "",
                f"- 用户初始诉求：{case.visible.get('user_request')}",
                f"- 标准结果：{case.expected_result['final_action']}，{case.expected_result['reason']}",
                f"- 组内平均 reward：{group['mean_reward']}",
                f"- 组内 reward 标准差：{group['reward_std']}",
                "",
                "| Rank | Policy | Reward | Advantage | Verdict | 主要惩罚 |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for item in group["trajectories"]:
            penalties = ", ".join(
                name for name, value in item["penalties"].items() if value
            ) or "无"
            lines.append(
                f"| {item['rank']} | {item['policy_name']} | {item['reward']} | "
                f"{item['advantage']} | {item['verdict']} | {penalties} |"
            )
        lines.extend(["", "**轨迹摘要**", ""])
        for trajectory in trajectories_by_case[group["case_id"]]:
            score = scores[trajectory["trajectory_id"]]
            lines.append(
                f"- `{trajectory['trajectory_id']}`：{trajectory['policy_name']}，"
                f"steps={len(trajectory['steps'])}，final={trajectory['final_decision']['type']}，"
                f"reward={score['reward']}。"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path

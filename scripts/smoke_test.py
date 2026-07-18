from __future__ import annotations

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gov_agent_rl.experiments import EXPERIMENTS, train_policy_mixture
from gov_agent_rl.grpo import build_grpo_groups
from gov_agent_rl.io_utils import load_cases, read_jsonl
from gov_agent_rl.rollout import rollout_cases, rollout_to_file
from gov_agent_rl.scoring import reward_profile, score_file, score_trajectory


CASES_DIR = Path("data/cases")


def main() -> None:
    cases = load_cases(CASES_DIR)
    trajectories = rollout_cases(cases)
    assert len(cases) >= 5
    assert len(trajectories) == len(cases) * 5

    subsidy = next(case for case in cases if case.case_id == "subsidy_001")
    careful = next(
        item for item in trajectories
        if item.case_id == "subsidy_001" and item.policy_name == "careful_policy"
    )
    assert careful.final_decision["type"] == "REFUSE"
    assert "MATERIAL_CHECK" in careful.steps[-1].state["tool_results"]

    hacker = next(
        item for item in trajectories
        if item.case_id == "subsidy_001" and item.policy_name == "judge_hacker_policy"
    )
    vulnerable = score_trajectory(subsidy, hacker.to_dict(), reward_profile("collapse_prone"))
    hardened = score_trajectory(subsidy, hacker.to_dict(), reward_profile("hardened"))
    assert vulnerable["reward"] > hardened["reward"]
    assert hardened["missing_tools"] == ["MATERIAL_CHECK", "RISK_CHECK"]

    with tempfile.TemporaryDirectory() as directory:
        out_dir = Path(directory)
        trajectories_path = rollout_to_file(CASES_DIR, out_dir)
        scores_path = score_file(CASES_DIR, trajectories_path, out_dir / "scored.jsonl")
        groups = build_grpo_groups(read_jsonl(scores_path))
        assert len(groups) == len(cases)
        assert all("reward_std" in group for group in groups)

    before = train_policy_mixture(CASES_DIR, EXPERIMENTS["before_fix"])[-1]
    after = train_policy_mixture(CASES_DIR, EXPERIMENTS["after_fix"])[-1]
    assert before["final_action_distribution"]["REFUSE"] > 0.90
    assert before["required_tool_rate"] < 0.10
    assert after["required_tool_rate"] > 0.70
    assert after["policy_entropy"] > before["policy_entropy"]
    print("smoke test passed")


if __name__ == "__main__":
    main()

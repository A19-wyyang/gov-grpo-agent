from pathlib import Path

from gov_agent_rl.grpo import build_grpo_groups
from gov_agent_rl.experiments import EXPERIMENTS, train_policy_mixture
from gov_agent_rl.io_utils import load_cases, read_jsonl
from gov_agent_rl.rollout import rollout_cases, rollout_to_file
from gov_agent_rl.policies import POLICIES
from gov_agent_rl.scoring import reward_profile, score_file, score_trajectory


CASES_DIR = Path("data/cases")


def test_rollout_records_recoverable_steps():
    cases = load_cases(CASES_DIR)
    trajectories = rollout_cases(cases)

    assert len(trajectories) == len(cases) * len(POLICIES)
    careful = [
        item for item in trajectories
        if item.case_id == "subsidy_001" and item.policy_name == "careful_policy"
    ][0]
    assert careful.final_decision["type"] == "REFUSE"
    assert careful.steps[0].state["known_slots"] == {"city": "杭州"}
    assert careful.steps[-1].slot_status["missing"] == []


def test_verifier_penalizes_bad_processes(tmp_path):
    trajectories_path = rollout_to_file(CASES_DIR, tmp_path)
    scores_path = score_file(CASES_DIR, trajectories_path, tmp_path / "scored.jsonl")
    scored = read_jsonl(scores_path)

    risky_subsidy = [
        row for row in scored
        if row["case_id"] == "subsidy_001" and row["policy_name"] == "risky_policy"
    ][0]
    over_refuse_business = [
        row for row in scored
        if row["case_id"] == "business_license_001"
        and row["policy_name"] == "over_refuse_policy"
    ][0]
    careful_business = [
        row for row in scored
        if row["case_id"] == "business_license_001" and row["policy_name"] == "careful_policy"
    ][0]

    assert risky_subsidy["penalties"]["early_submit"] == 1
    assert risky_subsidy["penalties"]["missing_required_tool"] == 1
    assert over_refuse_business["penalties"]["over_refuse"] == 1
    assert careful_business["reward"] > risky_subsidy["reward"]


def test_grpo_groups_rank_within_same_case(tmp_path):
    trajectories_path = rollout_to_file(CASES_DIR, tmp_path)
    scores_path = score_file(CASES_DIR, trajectories_path, tmp_path / "scored.jsonl")
    groups = build_grpo_groups(read_jsonl(scores_path))

    assert {group["case_id"] for group in groups} == {
        case.case_id for case in load_cases(CASES_DIR)
    }
    for group in groups:
        rewards = [item["reward"] for item in group["trajectories"]]
        ranks = [item["rank"] for item in group["trajectories"]]
        assert rewards == sorted(rewards, reverse=True)
        assert ranks == list(range(1, len(POLICIES) + 1))
        assert any(item["advantage"] > 0 for item in group["trajectories"])


def test_hardened_reward_penalizes_judge_hacking():
    cases = load_cases(CASES_DIR)
    subsidy = next(case for case in cases if case.case_id == "subsidy_001")
    hacker = next(
        item for item in rollout_cases(cases)
        if item.case_id == "subsidy_001" and item.policy_name == "judge_hacker_policy"
    )

    vulnerable = score_trajectory(
        subsidy, hacker.to_dict(), reward_profile("collapse_prone")
    )
    hardened = score_trajectory(subsidy, hacker.to_dict(), reward_profile("hardened"))

    assert vulnerable["reward"] > hardened["reward"]
    assert hardened["missing_tools"] == ["MATERIAL_CHECK", "RISK_CHECK"]


def test_entropy_bonus_experiment_recovers_tool_usage():
    before = train_policy_mixture(CASES_DIR, EXPERIMENTS["before_fix"])[-1]
    after = train_policy_mixture(CASES_DIR, EXPERIMENTS["after_fix"])[-1]

    assert before["final_action_distribution"]["REFUSE"] > 0.90
    assert before["required_tool_rate"] < 0.10
    assert after["required_tool_rate"] > 0.70
    assert after["policy_entropy"] > before["policy_entropy"]

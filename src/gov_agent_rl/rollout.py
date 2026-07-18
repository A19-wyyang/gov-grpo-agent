from __future__ import annotations

from pathlib import Path

from .environment import GovernmentServiceEnv
from .io_utils import load_cases, write_jsonl
from .models import GovCase, Trajectory, TrajectoryStep
from .policies import POLICIES, Policy


def rollout_case(
    case: GovCase,
    policy_name: str,
    policy: Policy,
    run_index: int,
    max_steps: int = 8,
) -> Trajectory:
    env = GovernmentServiceEnv(case, max_steps=max_steps)
    steps: list[TrajectoryStep] = []

    while not env.done:
        state_before = env.state.snapshot()
        action = policy(case, env.state, env.step_count + 1)
        observation = env.step(action)
        steps.append(
            TrajectoryStep(
                step=env.step_count,
                state=state_before,
                action=action.to_dict(),
                observation=observation,
                slot_status=env.slot_status(),
                failure_tags=list(env.state.failure_tags),
            )
        )

    final_decision = env.state.final_decision or {
        "type": "REFUSE",
        "message": "未能形成最终决策。",
    }
    return Trajectory(
        case_id=case.case_id,
        trajectory_id=f"{case.case_id}_run_{run_index:02d}_{policy_name}",
        policy_name=policy_name,
        steps=steps,
        final_decision=final_decision,
        failure_tags=list(env.state.failure_tags),
    )


def rollout_cases(
    cases: list[GovCase],
    policy_names: list[str] | None = None,
    max_steps: int = 8,
) -> list[Trajectory]:
    selected_names = policy_names or list(POLICIES)
    trajectories: list[Trajectory] = []
    run_index = 1
    for case in cases:
        for policy_name in selected_names:
            trajectories.append(
                rollout_case(
                    case=case,
                    policy_name=policy_name,
                    policy=POLICIES[policy_name],
                    run_index=run_index,
                    max_steps=max_steps,
                )
            )
            run_index += 1
    return trajectories


def rollout_to_file(
    cases_dir: Path,
    out_dir: Path,
    policy_names: list[str] | None = None,
    max_steps: int = 8,
) -> Path:
    cases = load_cases(cases_dir)
    trajectories = rollout_cases(cases, policy_names=policy_names, max_steps=max_steps)
    output = out_dir / "trajectories.jsonl"
    write_jsonl(output, (trajectory.to_dict() for trajectory in trajectories))
    return output

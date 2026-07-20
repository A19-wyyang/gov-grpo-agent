from __future__ import annotations

import json
from typing import Any

from .agent_env import GovernmentServiceEpisode
from .rewarding import score_episode
from .schema import CaseSpec

try:
    from verl.tools.base_tool import BaseTool, ToolResponse
except ImportError:  # Allows CPU unit tests without veRL.
    BaseTool = object  # type: ignore[assignment,misc]

    class ToolResponse:  # type: ignore[no-redef]
        def __init__(self, text: str | None = None):
            self.text = text


def _find_case(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        candidate = value.get("case")
        if isinstance(candidate, str):
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict) and "case_id" in decoded:
                return decoded
        if isinstance(candidate, dict) and "case_id" in candidate:
            return candidate
        if "case_id" in value and "hidden_truth" in value and "rules" in value:
            return value
        for nested in value.values():
            found = _find_case(nested)
            if found is not None:
                return found
    return None


class GovernmentServiceTool(BaseTool):
    """Stateful veRL tool: one isolated GovernmentServiceEpisode per rollout."""

    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.instances: dict[str, GovernmentServiceEpisode] = {}
        self.completed: dict[str, dict[str, Any]] = {}

    async def create(
        self, instance_id: str | None = None, **kwargs: Any
    ) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {})
        request_id = (
            create_kwargs.get("_agent_request_id")
            if isinstance(create_kwargs, dict)
            else None
        )
        if request_id and request_id in self.instances:
            episode = self.instances[request_id]
            return request_id, ToolResponse(
                text=json.dumps(episode.initial_observation(), ensure_ascii=False)
            )
        if instance_id is None:
            from uuid import uuid4

            instance_id = request_id or str(uuid4())
        case_data = _find_case(kwargs)
        if case_data is None:
            raise ValueError(
                "GovernmentServiceTool requires case data in extra_info.case or tools_kwargs"
            )
        episode = GovernmentServiceEpisode(
            CaseSpec.model_validate(case_data),
            max_steps=int(self.config.get("max_steps", 8)),
        )
        self.instances[instance_id] = episode
        return instance_id, ToolResponse(
            text=json.dumps(episode.initial_observation(), ensure_ascii=False)
        )

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict]:
        episode = self.instances[instance_id]
        observation = episode.execute(parameters)
        reward = 0.0
        if episode.done:
            breakdown = score_episode(episode)
            reward = breakdown.total
            self.completed[instance_id] = {
                "trajectory": episode.trajectory(),
                "reward": breakdown.model_dump(mode="json"),
            }
        metrics = {
            "done": float(episode.done),
            "rounds": float(episode.action_attempts),
            "illegal_action": float("illegal_action" in episode.failure_tags),
            "illegal_action_count": float(episode.failure_counts["illegal_action"]),
            "trailing_action_count": float(
                episode.failure_counts["action_after_done"]
            ),
        }
        return (
            ToolResponse(text=json.dumps(observation, ensure_ascii=False)),
            reward,
            metrics,
        )

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        episode = self.instances[instance_id]
        return score_episode(episode).total

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        episode = self.instances.get(instance_id)
        if episode is not None and episode.done:
            self.instances.pop(instance_id, None)
        if episode is not None and episode.done and instance_id not in self.completed:
            self.completed[instance_id] = {
                "trajectory": episode.trajectory(),
                "reward": score_episode(episode).model_dump(mode="json"),
            }

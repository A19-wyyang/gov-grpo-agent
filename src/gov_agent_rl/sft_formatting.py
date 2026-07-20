from __future__ import annotations

import json
from collections import Counter
from typing import Any


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "government_service",
        "description": "执行政务办理动作，完成信息、政策、资格、材料和风险核验。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "ASK_USER",
                        "POLICY_SEARCH",
                        "ELIGIBILITY_CHECK",
                        "MATERIAL_CHECK",
                        "RISK_CHECK",
                        "SUBMIT",
                        "REFUSE",
                    ],
                },
                "slot": {"type": "string"},
                "query": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


def serialize_sft_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Serialize a Qwen tool dialogue and return its assistant-only mask."""
    accumulated, mask, _ = serialize_sft_messages_with_turns(
        tokenizer, messages
    )
    return accumulated, mask


def serialize_sft_messages_with_turns(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    """Serialize dialogue with assistant mask and per-token assistant turn IDs."""
    if len(messages) < 2 or [message.get("role") for message in messages[:2]] != [
        "system",
        "user",
    ]:
        raise RuntimeError("SFT dialogue must start with system and user")
    prefix = tokenizer.apply_chat_template(
        messages[:2],
        tools=[TOOL_SCHEMA],
        tokenize=False,
        add_generation_prompt=False,
    )
    accumulated = tokenizer.encode(prefix, add_special_tokens=False)
    mask = [0] * len(accumulated)
    turn_ids = [-1] * len(accumulated)

    def append(text: str, trainable: bool, turn_id: int = -1) -> None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        accumulated.extend(token_ids)
        mask.extend([1 if trainable else 0] * len(token_ids))
        turn_ids.extend(
            [turn_id if trainable else -1] * len(token_ids)
        )

    assistant_turn = 0
    for message in messages[2:]:
        role = message.get("role")
        if role == "assistant":
            append("<|im_start|>assistant\n", False)
            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    function = call["function"]
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    payload = {
                        "name": function["name"],
                        "arguments": arguments,
                    }
                    append(
                        "<tool_call>\n"
                        + json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n</tool_call>",
                        True,
                        assistant_turn,
                    )
            else:
                append(
                    str(message.get("content", "")),
                    True,
                    assistant_turn,
                )
            append("<|im_end|>\n", True, assistant_turn)
            assistant_turn += 1
        elif role == "tool":
            append("<|im_start|>user\n<tool_response>\n", False)
            append(str(message.get("content", "")), False)
            append("\n</tool_response><|im_end|>\n", False)
        else:
            raise RuntimeError(f"unexpected role after prefix: {role}")
    return accumulated, mask, turn_ids


def enforce_sft_max_length(token_count: int, max_length: int) -> None:
    if token_count > max_length:
        raise RuntimeError(
            f"SFT sequence has {token_count} tokens, exceeding "
            f"max_length={max_length}; increase --max-length or shorten the "
            "source trajectory. Tail truncation is disabled because it can "
            "remove final verification and decision targets."
        )


def turn_balanced_weights(
    assistant_mask: list[int],
    turn_ids: list[int],
) -> list[float]:
    """Assign equal total loss mass to every assistant decision turn."""
    if len(assistant_mask) != len(turn_ids):
        raise ValueError("assistant mask and turn IDs must have equal length")
    counts = Counter(
        turn_id
        for mask, turn_id in zip(assistant_mask, turn_ids, strict=True)
        if int(mask) == 1 and turn_id >= 0
    )
    if not counts:
        raise ValueError("sample contains no assistant decision turn")
    return [
        0.0
        if int(mask) == 0
        else 1.0 / counts[turn_id]
        for mask, turn_id in zip(assistant_mask, turn_ids, strict=True)
    ]

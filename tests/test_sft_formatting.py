import pytest

from gov_agent_rl.data_builder import build_cases, build_sft_messages
from gov_agent_rl.sft_formatting import (
    enforce_sft_max_length,
    serialize_sft_messages,
    serialize_sft_messages_with_turns,
    turn_balanced_weights,
)


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "".join(
            f"<{message['role']}>{message.get('content', '')}"
            for message in messages
        )

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_sft_serialization_masks_only_assistant_targets():
    case = build_cases()[0]
    messages = build_sft_messages(case)
    input_ids, mask = serialize_sft_messages(CharacterTokenizer(), messages)
    assert len(input_ids) == len(mask)
    assert any(mask)
    assert any(value == 0 for value in mask)
    rendered_targets = "".join(
        chr(token) for token, trainable in zip(input_ids, mask) if trainable
    )
    assert "<tool_call>" in rendered_targets
    assert case.expected_result.reason in rendered_targets


def test_sft_max_length_fails_instead_of_truncating_tail():
    enforce_sft_max_length(token_count=4096, max_length=4096)
    with pytest.raises(RuntimeError, match="Tail truncation is disabled"):
        enforce_sft_max_length(token_count=4097, max_length=4096)


def test_turn_balanced_weights_give_each_assistant_decision_equal_mass():
    case = build_cases()[0]
    messages = build_sft_messages(case)
    input_ids, mask, turn_ids = serialize_sft_messages_with_turns(
        CharacterTokenizer(), messages
    )
    weights = turn_balanced_weights(mask, turn_ids)
    assert len(input_ids) == len(mask) == len(turn_ids) == len(weights)
    assistant_turns = sorted({turn_id for turn_id in turn_ids if turn_id >= 0})
    assert len(assistant_turns) == sum(
        message["role"] == "assistant" for message in messages
    )
    for turn_id in assistant_turns:
        mass = sum(
            weight
            for weight, token_turn in zip(weights, turn_ids, strict=True)
            if token_turn == turn_id
        )
        assert mass == pytest.approx(1.0)
    assert all(
        weight == 0.0
        for weight, trainable in zip(weights, mask, strict=True)
        if not trainable
    )

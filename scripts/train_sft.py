from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.sft.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sft-qwen3-8b"))
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None = None) -> Dataset:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]
    return Dataset.from_list(rows)


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_rows: list[dict[str, list[int]]] = []
        for example in examples:
            input_ids, assistant_mask = self._delta_tokenize(example["messages"])
            attention_mask = [1] * len(input_ids)
            labels = [
                token if int(mask) == 1 else -100
                for token, mask in zip(input_ids, assistant_mask, strict=True)
            ]
            if not any(label != -100 for label in labels):
                raise RuntimeError("sample contains no assistant training tokens")
            encoded_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        max_len = max(len(row["input_ids"]) for row in encoded_rows)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in encoded_rows:
            padding = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    def _delta_tokenize(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[int], list[int]]:
        """Serialize Qwen3 tool dialogue and mask assistant-generated tokens.

        Qwen3 rewrites earlier messages when a tool response is appended, so
        whole-conversation prefix differencing is unsafe. We let its template
        render the tool-aware system/user prefix, then append the documented
        Qwen tool-call wire format in immutable pieces. Assistant role markers
        are context; tool-call/content and ``im_end`` are the learning target.
        """
        if len(messages) < 2 or [m.get("role") for m in messages[:2]] != [
            "system",
            "user",
        ]:
            raise RuntimeError("SFT dialogue must start with system and user")
        prefix = self.tokenizer.apply_chat_template(
            messages[:2],
            tools=[TOOL_SCHEMA],
            tokenize=False,
            add_generation_prompt=False,
        )
        accumulated = self.tokenizer.encode(prefix, add_special_tokens=False)
        mask = [0] * len(accumulated)

        def append(text: str, trainable: bool) -> None:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            accumulated.extend(token_ids)
            mask.extend([1 if trainable else 0] * len(token_ids))

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
                            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                            + "\n</tool_call>",
                            True,
                        )
                else:
                    append(str(message.get("content", "")), True)
                append("<|im_end|>\n", True)
            elif role == "tool":
                append("<|im_start|>user\n<tool_response>\n", False)
                append(str(message.get("content", "")), False)
                append("\n</tool_response><|im_end|>\n", False)
            else:
                raise RuntimeError(f"unexpected role after prefix: {role}")
        if len(accumulated) > self.max_length:
            accumulated = accumulated[: self.max_length]
            mask = mask[: self.max_length]
        return accumulated, mask


def main() -> None:
    args = parse_args()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    if local_rank == 0:
        model.print_trainable_parameters()

    limit = 16 if args.smoke else None
    train_dataset = load_jsonl(args.train, limit)
    eval_dataset = load_jsonl(args.validation, 8 if args.smoke else None)
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=0.05 if args.smoke else args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.gradient_accumulation,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps",
        eval_steps=4 if args.smoke else 50,
        save_strategy="steps",
        save_steps=4 if args.smoke else 50,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        # A one-step smoke run must use a non-zero LR; ratio warmup makes its
        # only optimizer step land at LR=0 and therefore does not test updates.
        warmup_steps=0 if args.smoke else 10,
        lr_scheduler_type="cosine",
        seed=42,
        data_seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer, args.max_length),
    )
    trainer.train()
    trainer.save_model(str(args.output / "final_adapter"))
    tokenizer.save_pretrained(str(args.output / "final_adapter"))


if __name__ == "__main__":
    main()

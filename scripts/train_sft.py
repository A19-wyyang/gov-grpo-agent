from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from gov_agent_rl.sft_formatting import (
    enforce_sft_max_length,
    serialize_sft_messages,
    serialize_sft_messages_with_turns,
    turn_balanced_weights,
)
from gov_agent_rl.sft_reporting import (
    render_scenario_losses,
    render_sft_training,
)


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
    parser.add_argument("--turn-balanced-loss", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=25)
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
    turn_balanced_loss: bool = False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_rows: list[dict[str, list[int]]] = []
        for example in examples:
            input_ids, assistant_mask, turn_ids = self._delta_tokenize(
                example["messages"]
            )
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
                    "loss_weights": (
                        turn_balanced_weights(assistant_mask, turn_ids)
                        if self.turn_balanced_loss
                        else []
                    ),
                }
            )

        max_len = max(len(row["input_ids"]) for row in encoded_rows)
        pad_id = self.tokenizer.pad_token_id
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        if self.turn_balanced_loss:
            batch["loss_weights"] = []
        for row in encoded_rows:
            padding = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
            if self.turn_balanced_loss:
                batch["loss_weights"].append(
                    row["loss_weights"] + [0.0] * padding
                )
        return {
            key: torch.tensor(
                value,
                dtype=torch.float32 if key == "loss_weights" else torch.long,
            )
            for key, value in batch.items()
        }

    def _delta_tokenize(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[int], list[int], list[int]]:
        accumulated, mask, turn_ids = serialize_sft_messages_with_turns(
            self.tokenizer, messages
        )
        enforce_sft_max_length(len(accumulated), self.max_length)
        return accumulated, mask, turn_ids

    def _serialize_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[int], list[int]]:
        """Serialize Qwen3 tool dialogue and mask assistant-generated tokens.

        Qwen3 rewrites earlier messages when a tool response is appended, so
        whole-conversation prefix differencing is unsafe. We let its template
        render the tool-aware system/user prefix, then append the documented
        Qwen tool-call wire format in immutable pieces. Assistant role markers
        are context; tool-call/content and ``im_end`` are the learning target.
        """
        return serialize_sft_messages(self.tokenizer, messages)


class TurnBalancedTrainer(Trainer):
    """Average token CE within each assistant turn, then across turns."""

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Any:
        del num_items_in_batch
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid = shift_labels.ne(-100)
        weights = shift_weights * valid
        denominator = weights.sum().clamp_min(
            torch.finfo(weights.dtype).eps
        )
        loss = (token_loss * weights).sum() / denominator
        return (loss, outputs) if return_outputs else loss


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def audit_sft_dataset(
    dataset: Dataset,
    collator: AssistantOnlyCollator,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, int, int, float]]] = defaultdict(list)
    overlong: list[dict[str, Any]] = []
    for example in dataset:
        input_ids, assistant_mask, turn_ids = serialize_sft_messages_with_turns(
            collator.tokenizer, example["messages"]
        )
        scenario = str(example.get("scenario_type", "unknown"))
        token_count = len(input_ids)
        assistant_tokens = sum(int(value) for value in assistant_mask)
        turn_counts: dict[int, int] = defaultdict(int)
        for mask, turn_id in zip(assistant_mask, turn_ids, strict=True):
            if int(mask) == 1:
                turn_counts[turn_id] += 1
        imbalance = max(turn_counts.values()) / min(turn_counts.values())
        grouped[scenario].append(
            (token_count, assistant_tokens, len(turn_counts), imbalance)
        )
        if token_count > collator.max_length:
            overlong.append(
                {
                    "case_id": str(example.get("case_id", "")),
                    "scenario_type": scenario,
                    "tokens": token_count,
                    "assistant_tokens": assistant_tokens,
                }
            )

    def summarize(
        rows: list[tuple[int, int, int, float]]
    ) -> dict[str, Any]:
        tokens = [row[0] for row in rows]
        targets = [row[1] for row in rows]
        turns = [row[2] for row in rows]
        imbalances = [row[3] for row in rows]
        return {
            "count": len(rows),
            "tokens_mean": round(mean(tokens), 2),
            "tokens_p50": _percentile(tokens, 0.50),
            "tokens_p95": _percentile(tokens, 0.95),
            "tokens_max": max(tokens),
            "assistant_tokens_mean": round(mean(targets), 2),
            "assistant_tokens_min": min(targets),
            "assistant_turns_mean": round(mean(turns), 2),
            "turn_token_imbalance_mean": round(mean(imbalances), 4),
        }

    all_rows = [row for rows in grouped.values() for row in rows]
    return {
        "max_length": collator.max_length,
        "loss_mode": (
            "turn_balanced" if collator.turn_balanced_loss else "token_mean"
        ),
        "overall": summarize(all_rows),
        "by_scenario": {
            scenario: summarize(rows) for scenario, rows in sorted(grouped.items())
        },
        "overlong_count": len(overlong),
        "overlong_examples": sorted(
            overlong, key=lambda row: int(row["tokens"]), reverse=True
        )[:20],
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    limit = 16 if args.smoke else None
    train_dataset = load_jsonl(args.train, limit)
    eval_dataset = load_jsonl(args.validation, 8 if args.smoke else None)
    args.output.mkdir(parents=True, exist_ok=True)
    collator = AssistantOnlyCollator(
        tokenizer,
        args.max_length,
        turn_balanced_loss=args.turn_balanced_loss,
    )
    data_audit = {
        "train": audit_sft_dataset(train_dataset, collator),
        "validation": audit_sft_dataset(eval_dataset, collator),
    }
    if local_rank == 0:
        (args.output / "sft_data_audit.json").write_text(
            json.dumps(data_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    overlong_count = sum(
        int(split["overlong_count"]) for split in data_audit.values()
    )
    if overlong_count:
        raise RuntimeError(
            f"{overlong_count} SFT sequences exceed max_length={args.max_length}; "
            f"see {args.output / 'sft_data_audit.json'}"
        )

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
        eval_steps=4 if args.smoke else args.eval_steps,
        save_strategy="steps",
        save_steps=4 if args.smoke else args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
    trainer_class = TurnBalancedTrainer if args.turn_balanced_loss else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    if not args.smoke and trainer.state.best_model_checkpoint is None:
        raise RuntimeError(
            "formal SFT produced no evaluated checkpoint; lower "
            "--eval-steps or increase --epochs"
        )
    trainer.save_model(str(args.output / "final_adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(args.output / "final_adapter"))

    scenario_eval: dict[str, dict[str, float]] = {}
    for scenario in sorted(set(eval_dataset["scenario_type"])):
        scenario_dataset = eval_dataset.filter(
            lambda example: example["scenario_type"] == scenario
        )
        metrics = trainer.evaluate(
            eval_dataset=scenario_dataset,
            metric_key_prefix=f"eval_{scenario}",
        )
        scenario_eval[scenario] = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, Real)
        }
    if trainer.is_world_process_zero():
        log_history = list(trainer.state.log_history)
        (args.output / "training_log.json").write_text(
            json.dumps(log_history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output / "scenario_eval_metrics.json").write_text(
            json.dumps(scenario_eval, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {
            "loss_mode": (
                "turn_balanced"
                if args.turn_balanced_loss
                else "token_mean"
            ),
            "global_step": trainer.state.global_step,
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "eval_steps": args.eval_steps,
        }
        (args.output / "sft_training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        render_sft_training(
            log_history,
            args.output / "sft_training_metrics.png",
        )
        render_scenario_losses(
            scenario_eval,
            args.output / "sft_scenario_metrics.png",
        )


if __name__ == "__main__":
    main()

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SftTrainingConfig:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    train_file: str = "artifacts/mvp/sft_samples.jsonl"
    output_dir: str = "artifacts/qwen3_8b_sft_lora"
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_steps: int = 100
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True


def format_chatml_sample(sample):
    parts = []
    for message in sample["messages"]:
        parts.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        )
    return "".join(parts)


def load_sft_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record["text"] = format_chatml_sample(record)
            records.append(record)
    return records


def train(config):
    # Heavy training dependencies are imported lazily so tests and data generation
    # still run on machines without GPU training stacks installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    records = load_sft_jsonl(config.train_file)
    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return {"output_dir": config.output_dir, "train_samples": len(records)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Qwen3 government agent actions.")
    parser.add_argument("--model-name-or-path", default=SftTrainingConfig.model_name_or_path)
    parser.add_argument("--train-file", default=SftTrainingConfig.train_file)
    parser.add_argument("--output-dir", default=SftTrainingConfig.output_dir)
    parser.add_argument("--max-seq-length", type=int, default=SftTrainingConfig.max_seq_length)
    parser.add_argument("--per-device-train-batch-size", type=int, default=SftTrainingConfig.per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=SftTrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--learning-rate", type=float, default=SftTrainingConfig.learning_rate)
    parser.add_argument("--num-train-epochs", type=float, default=SftTrainingConfig.num_train_epochs)
    parser.add_argument("--lora-rank", type=int, default=SftTrainingConfig.lora_rank)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = SftTrainingConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lora_rank=args.lora_rank,
        load_in_4bit=not args.no_4bit,
    )
    result = train(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

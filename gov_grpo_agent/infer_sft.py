import argparse
import json
from pathlib import Path

from gov_grpo_agent.schemas import validate_action
from gov_grpo_agent.train_sft import configure_4090_nccl_environment


SYSTEM_PROMPT = (
    "你是政务办理Agent。必须只输出一个合法JSON动作，格式为"
    '{"action": "...", "arguments": {...}}。'
    "不要输出解释、思考过程或Markdown。"
)


def build_inference_prompt(user_query):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_action_from_generation(text):
    cleaned = text.replace("<|im_end|>", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in generation: {text!r}")
    payload = cleaned[start : end + 1]
    action = json.loads(payload)
    validate_action(action)
    return action


def generate_action(model_name_or_path, adapter_path, user_query, max_new_tokens=256):
    configure_4090_nccl_environment()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    prompt = build_inference_prompt(user_query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return {
        "user_query": user_query,
        "raw_generation": generated_text,
        "action": parse_action_from_generation(generated_text),
    }


class SftActionGenerator:
    def __init__(self, model_name_or_path, adapter_path, max_new_tokens=256):
        configure_4090_nccl_environment()
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.torch = torch

    def generate(self, prompt):
        inference_prompt = build_inference_prompt(prompt)
        inputs = self.tokenizer(inference_prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return parse_action_from_generation(generated_text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate Qwen SFT LoRA action generation.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter-path", default="artifacts/qwen3_8b_sft_lora")
    parser.add_argument("--query", default="我想提取公积金交房租，应该怎么办？")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    result = generate_action(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        user_query=args.query,
        max_new_tokens=args.max_new_tokens,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

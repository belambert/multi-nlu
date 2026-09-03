"""LoRA fine-tuning of a causal LM to emit inline-XML annotations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerBase, Trainer, TrainingArguments

from multi_nlu.data import Example
from multi_nlu.models import load_model
from multi_nlu.prompts import build_messages
from multi_nlu.schema import Schema

IGNORE = -100


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen3-1.7B"
    out_dir: Path = Path("runs/latest")
    epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    warmup_steps: int = 50
    max_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    eval_steps: int = 200
    log_steps: int = 20
    seed: int = 0
    device: str | None = field(default=None)


def train(
    schema: Schema,
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    config: TrainConfig,
) -> Path:
    """Fine-tune a LoRA adapter and save it to config.out_dir."""
    model, tokenizer, device = load_model(config.model_id, device=config.device, train=True)
    tokenizer.padding_side = "right"

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(config.out_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=config.log_steps,
        eval_strategy="steps" if eval_examples else "no",
        eval_steps=config.eval_steps,
        save_strategy="no",
        bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        report_to=[],
        seed=config.seed,
    )

    encode = Encoder(tokenizer, schema, config.max_len)
    Trainer(
        model=model,
        args=args,
        train_dataset=[encode(e) for e in train_examples],
        eval_dataset=[encode(e) for e in eval_examples] or None,
        data_collator=collate_padded(tokenizer),
    ).train()

    model.save_pretrained(str(config.out_dir))
    tokenizer.save_pretrained(str(config.out_dir))
    return config.out_dir


@dataclass
class Encoder:
    """Turns an example into token ids whose prompt half is masked out of the loss."""

    tokenizer: PreTrainedTokenizerBase
    schema: Schema
    max_len: int

    def __call__(self, example: Example) -> dict[str, list[int]]:
        messages = build_messages(example.text, self.schema)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(
            example.xml + self.tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]

        ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([IGNORE] * len(prompt_ids) + answer_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate_padded(tokenizer: PreTrainedTokenizerBase):
    """Right-pad a batch, padding labels with the ignore index."""

    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        pad = tokenizer.pad_token_id

        def stack(key: str, fill: int) -> torch.Tensor:
            rows = [f[key] + [fill] * (width - len(f[key])) for f in features]
            return torch.tensor(rows, dtype=torch.long)

        return {
            "input_ids": stack("input_ids", pad),
            "labels": stack("labels", IGNORE),
            "attention_mask": torch.tensor(
                [[1] * len(f["input_ids"]) + [0] * (width - len(f["input_ids"])) for f in features],
                dtype=torch.long,
            ),
        }

    return collate

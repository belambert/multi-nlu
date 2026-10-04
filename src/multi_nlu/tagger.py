"""A shared encoder with two token-classification heads: intent spans and slots.

Any Hugging Face backbone loadable with `AutoModel` works, so a decoder like
Qwen can be compared with an encoder like ModernBERT without code changes; a
decoder only sees its left context, which is part of what that comparison tests.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from multi_nlu.annotation import Annotation
from multi_nlu.data import Example
from multi_nlu.models import pick_device
from multi_nlu.schema import Schema
from multi_nlu.tagging import IGNORE, Labels, decode, encode

LABELS_FILE = "labels.json"
WEIGHTS_FILE = "model.safetensors"


@dataclass
class TagConfig:
    model_id: str = "answerdotai/ModernBERT-base"
    out_dir: Path = Path("runs/tagger")
    epochs: float = 3.0
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    slot_weight: float = 1.0
    dropout: float = 0.1
    max_len: int = 128
    eval_steps: int = 500
    log_steps: int = 50
    seed: int = 0
    device: str | None = field(default=None)
    wandb: bool = True
    wandb_project: str = "multi-nlu"


class Tagger(nn.Module):
    """Backbone hidden states feed one linear head per tag layer."""

    encoder: PreTrainedModel

    def __init__(
        self,
        encoder: PreTrainedModel,
        labels: Labels,
        dropout: float = 0.1,
        slot_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder, self.labels, self.slot_weight = encoder, labels, slot_weight
        hidden = encoder.config.get_text_config().hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, len(labels.intents))
        self.slot_head = nn.Linear(hidden, len(labels.slots))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: torch.Tensor | None = None,
        slot_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        out = {"intent_logits": self.intent_head(hidden), "slot_logits": self.slot_head(hidden)}

        if intent_labels is not None and slot_labels is not None:
            out["loss"] = _ce(out["intent_logits"], intent_labels) + self.slot_weight * _ce(
                out["slot_logits"], slot_labels
            )
        return out


def train(
    schema: Schema,
    train_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    config: TagConfig,
) -> Path:
    """Fine-tune backbone and heads together, saving the tagger to config.out_dir."""
    import wandb

    device = pick_device(config.device)
    tok = load_tokenizer(config.model_id)
    labels = Labels.from_schema(schema)
    # fp32 weights to match the heads; on cuda, Trainer's bf16 autocast covers speed
    encoder = AutoModel.from_pretrained(config.model_id, dtype=torch.float32)
    model = Tagger(encoder, labels, config.dropout, config.slot_weight).to(device)

    if config.wandb:
        hyper = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
        wandb.init(project=config.wandb_project, config=hyper, name=config.out_dir.name)

    args = TrainingArguments(
        output_dir=str(config.out_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=config.log_steps,
        eval_strategy="steps" if eval_examples else "no",
        eval_steps=config.eval_steps,
        save_strategy="no",
        label_names=["intent_labels", "slot_labels"],
        prediction_loss_only=True,
        bf16=device.type == "cuda" and torch.cuda.is_bf16_supported(),
        use_cpu=device.type == "cpu",
        report_to="wandb" if config.wandb else "none",
        seed=config.seed,
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=features(train_examples, tok, labels, config.max_len),
        eval_dataset=features(eval_examples, tok, labels, config.max_len) or None,
        data_collator=collate(tok),
    ).train()

    save(model, tok, config.out_dir)
    return config.out_dir


@torch.inference_mode()
def predict(
    texts: Sequence[str],
    model: Tagger,
    tok: PreTrainedTokenizerBase,
    schema: Schema | None = None,
    batch_size: int = 64,
    max_len: int = 128,
    progress: bool = True,
) -> list[Annotation]:
    """Tag each utterance, taking the most likely tag per token in each layer."""
    device = next(model.parameters()).device
    out: list[Annotation] = []

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in tqdm(batches, desc="tag", disable=not progress):
        enc = tok(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        out.extend(
            decode(text, offsets, intents, slots, model.labels, schema)
            for text, offsets, intents, slots in zip(
                batch,
                enc["offset_mapping"].tolist(),
                logits["intent_logits"].argmax(-1).tolist(),
                logits["slot_logits"].argmax(-1).tolist(),
            )
        )
    return out


def save(model: Tagger, tok: PreTrainedTokenizerBase, out_dir: Path) -> None:
    """Backbone config, tokenizer, both label vocabularies and all weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / LABELS_FILE).write_text(model.labels.to_json())
    save_model(model, str(out_dir / WEIGHTS_FILE))


def load(path: Path, device: str | None = None) -> tuple[Tagger, PreTrainedTokenizerBase]:
    """Load a saved tagger for inference."""
    encoder: Any = AutoModel.from_config(AutoConfig.from_pretrained(str(path)))
    model = Tagger(encoder, Labels.from_json((path / LABELS_FILE).read_text()))
    load_model(model, str(path / WEIGHTS_FILE), device="cpu")
    return model.to(pick_device(device)).eval(), load_tokenizer(str(path))


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def features(
    examples: Sequence[Example], tok: PreTrainedTokenizerBase, labels: Labels, max_len: int
) -> list[dict[str, list[int]]]:
    """Token ids and both tag layers for each example."""
    if not examples:
        return []

    enc = tok(
        [e.text for e in examples], truncation=True, max_length=max_len, return_offsets_mapping=True
    )
    rows = []
    for example, ids, offsets in zip(examples, enc["input_ids"], enc["offset_mapping"]):
        intent_ids, slot_ids = encode(example.annotation, offsets, labels)
        rows.append({"input_ids": ids, "intent_labels": intent_ids, "slot_labels": slot_ids})
    return rows


def collate(tok: PreTrainedTokenizerBase):
    """Right-pad a batch, padding both tag layers with the ignore index."""

    def run(rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(r["input_ids"]) for r in rows)

        def stack(key: str, fill: int) -> torch.Tensor:
            return torch.tensor([r[key] + [fill] * (width - len(r[key])) for r in rows])

        return {
            "input_ids": stack("input_ids", tok.pad_token_id),
            "attention_mask": torch.tensor(
                [[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in rows]
            ),
            "intent_labels": stack("intent_labels", IGNORE),
            "slot_labels": stack("slot_labels", IGNORE),
        }

    return run


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)

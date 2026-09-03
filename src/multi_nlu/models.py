"""Loading models and tokenizers across cpu, mps and cuda."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, train: bool = False) -> torch.dtype:
    """Half precision where it is safe; cpu and mps train in fp32 to stay stable."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps" and not train:
        return torch.float16
    return torch.float32


def load_model(
    model_id: str,
    adapter: str | None = None,
    device: str | None = None,
    train: bool = False,
) -> tuple[Any, PreTrainedTokenizerBase, torch.device]:
    """Load a causal LM, optionally with a LoRA adapter merged in for inference."""
    dev = pick_device(device)
    model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=pick_dtype(dev, train))

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if not train:
            model = model.merge_and_unload()

    model.to(dev)
    model.train(train)

    return model, load_tokenizer(model_id), dev


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Tokenizer set up for left-padded batched generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

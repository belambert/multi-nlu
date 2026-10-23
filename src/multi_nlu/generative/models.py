"""Loading causal LMs and their tokenizers, optionally with a LoRA adapter."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from multi_nlu.device import pick_device, pick_dtype


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

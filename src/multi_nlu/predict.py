"""Batched tagging of utterances with a local model."""

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from multi_nlu.annotation import Annotation, parse_xml
from multi_nlu.data import Example
from multi_nlu.prompts import build_messages
from multi_nlu.schema import Schema


@dataclass(frozen=True)
class Prediction:
    text: str
    xml: str
    gold: str | None = None

    @property
    def annotation(self) -> Annotation:
        return parse_xml(self.xml, self.text)


@torch.inference_mode()
def predict(
    texts: Sequence[str],
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    schema: Schema,
    shots: Sequence[Example] = (),
    batch_size: int = 8,
    max_new_tokens: int | None = None,
    progress: bool = True,
) -> list[Prediction]:
    """Tag each utterance, returning raw generations alongside their input."""
    device = next(model.parameters()).device
    out: list[Prediction] = []

    batches = list(_batched(texts, batch_size))
    for batch in tqdm(batches, desc="predict", disable=not progress):
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(text, schema, shots),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or _budget(batch, tokenizer),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        out.extend(Prediction(text, clean(xml)) for text, xml in zip(batch, completions))

    return out


def clean(completion: str) -> str:
    """Strip the code fences and preamble that chat models like to add."""
    text = completion.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("xml").removeprefix("html")
    start = text.find("<")
    return text[start:].strip() if start >= 0 else text


def _budget(texts: Sequence[str], tokenizer: PreTrainedTokenizerBase) -> int:
    """Tagging roughly triples the token count; leave headroom on top of that."""
    longest = max(len(tokenizer(t)["input_ids"]) for t in texts)
    return 4 * longest + 64


def _batched(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

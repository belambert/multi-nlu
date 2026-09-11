"""Batched annotation of utterances with a local model."""

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from multi_nlu.annotation import Annotation
from multi_nlu.data import Example
from multi_nlu.formats import DEFAULT, Format, get_format
from multi_nlu.prompts import build_messages
from multi_nlu.schema import Schema


@dataclass(frozen=True)
class Prediction:
    """One generation, with the format needed to read it back and its gold target."""

    text: str
    output: str
    gold: str | None = None
    format: str = DEFAULT

    @property
    def annotation(self) -> Annotation:
        return get_format(self.format).parse(self.output, self.text)

    @property
    def gold_annotation(self) -> Annotation:
        if self.gold is None:
            raise ValueError("prediction carries no gold target")
        return get_format(self.format).parse(self.gold, self.text)


@torch.inference_mode()
def predict(
    texts: Sequence[str],
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    schema: Schema,
    fmt: Format,
    shots: Sequence[Example] = (),
    batch_size: int = 8,
    max_new_tokens: int | None = None,
    progress: bool = True,
) -> list[Prediction]:
    """Annotate each utterance, returning cleaned generations alongside their input."""
    device = next(model.parameters()).device
    out: list[Prediction] = []

    batches = list(_batched(texts, batch_size))
    for batch in tqdm(batches, desc="predict", disable=not progress):
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(text, schema, fmt, shots),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for text in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or _budget(batch, tokenizer, fmt),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = tokenizer.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        out.extend(
            Prediction(text, fmt.clean(completion), format=fmt.name)
            for text, completion in zip(batch, completions)
        )

    return out


def _budget(texts: Sequence[str], tokenizer: PreTrainedTokenizerBase, fmt: Format) -> int:
    """Annotating inflates the token count; leave headroom on top of that."""
    longest = max(len(tokenizer(t)["input_ids"]) for t in texts)
    return fmt.budget_factor * longest + 64


def _batched(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

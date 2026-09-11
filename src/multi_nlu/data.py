"""Loading annotated utterances from a schema's dataset."""

import random
from dataclasses import dataclass
from typing import Any, Sequence

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.schema import DatasetSpec, Schema


@dataclass(frozen=True)
class Example:
    """A gold example, held format-neutrally; each format renders its own target."""

    text: str
    annotation: Annotation


def load_examples(
    source: Schema | DatasetSpec, split: str, limit: int | None = None, seed: int = 0
) -> list[Example]:
    """Load one split; a limit takes a fixed random sample, not a prefix."""
    from datasets import load_dataset

    spec = source.dataset if isinstance(source, Schema) else source
    if spec is None:
        raise ValueError(f"schema {source.name!r} declares no dataset to load")

    data = load_dataset(spec.path, name=spec.name, split=split)
    if limit is not None and limit < len(data):
        data = data.shuffle(seed=seed).select(range(limit))

    return [
        Example(r[spec.text_column], to_annotation(r[spec.text_column], r[spec.intents_column]))
        for r in data
    ]


def to_annotation(text: str, intents: Sequence[dict[str, Any]]) -> Annotation:
    """Build an annotation from a dataset row's offset-annotated intents."""
    return Annotation(
        text=text,
        intents=tuple(
            IntentSpan(
                i["intent"],
                i["start"],
                i["end"],
                tuple(Span(s["name"], s["start"], s["end"]) for s in i["slots"]),
            )
            for i in intents
        ),
    )


def sample_shots(examples: Sequence[Example], k: int, seed: int = 0) -> list[Example]:
    """Pick k few-shot demonstrations, spreading them over intent counts."""
    if k >= len(examples):
        return list(examples)

    by_count: dict[int, list[Example]] = {}
    for example in examples:
        by_count.setdefault(len(example.annotation.intents), []).append(example)

    rng = random.Random(seed)
    pools = [rng.sample(v, len(v)) for _, v in sorted(by_count.items())]
    shots: list[Example] = []
    while len(shots) < k:
        for pool in pools:
            if pool and len(shots) < k:
                shots.append(pool.pop())
    return shots

"""Loading annotated utterances from a schema's dataset."""

import random
from dataclasses import dataclass
from typing import Sequence

from multi_nlu.annotation import Annotation, parse_xml
from multi_nlu.schema import DatasetSpec, Schema


@dataclass(frozen=True)
class Example:
    text: str
    xml: str

    @property
    def annotation(self) -> Annotation:
        return parse_xml(self.xml, self.text)


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

    return [Example(r[spec.text_column], r[spec.target_column]) for r in data]


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

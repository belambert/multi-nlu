"""Intent and slot spans as two layers of BIO tags over subword tokens.

Intents never nest and slots sit inside intents, so the whole annotation fits in
two flat tag sequences: one over intent labels, one over slot labels. Decoding
reads both back as spans and puts each slot under the intent it falls in.
"""

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.schema import Schema

IGNORE = -100
O = "O"

Offset = tuple[int, int] | Sequence[int]


@dataclass(frozen=True)
class Labels:
    """The two tag vocabularies, intent and slot, each `O` plus `B-`/`I-` per label."""

    intents: tuple[str, ...]
    slots: tuple[str, ...]

    @classmethod
    def from_schema(cls, schema: Schema) -> "Labels":
        return cls(_bio(schema.intents), _bio(schema.slots))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Labels":
        doc = json.loads(text)
        return cls(tuple(doc["intents"]), tuple(doc["slots"]))


def encode(
    annotation: Annotation, offsets: Sequence[Offset], labels: Labels
) -> tuple[list[int], list[int]]:
    """Tag ids for each token, intent layer then slot layer; textless tokens get IGNORE."""
    trimmed = [_trim(annotation.text, o) for o in offsets]
    return (
        _tag(annotation.intents, trimmed, labels.intents),
        _tag(annotation.slots, trimmed, labels.slots),
    )


def decode(
    text: str,
    offsets: Sequence[Offset],
    intent_ids: Sequence[int],
    slot_ids: Sequence[int],
    labels: Labels,
    schema: Schema | None = None,
) -> Annotation:
    """Read both tag layers back into an annotation, keeping only slots the schema
    allows under their intent when a schema is given."""
    trimmed = [_trim(text, o) for o in offsets]
    intents = _spans(trimmed, _names(labels.intents, intent_ids))
    slots = _spans(trimmed, _names(labels.slots, slot_ids))
    if not intents:
        return Annotation(text)

    owned: list[list[Span]] = [[] for _ in intents]
    for slot in slots:
        owned[_owner(slot, intents)].append(slot)
    return Annotation(text, tuple(_nest(i, s, schema) for i, s in zip(intents, owned)))


def _bio(names: Iterable[str]) -> tuple[str, ...]:
    return (O, *(f"{prefix}-{name}" for name in names for prefix in "BI"))


def _names(vocab: Sequence[str], ids: Sequence[int]) -> list[str]:
    """Tag names for ids; IGNORE reads as O, so gold tags decode as well as predictions."""
    return [O if i == IGNORE else vocab[i] for i in ids]


def _trim(text: str, offset: Offset) -> tuple[int, int]:
    """A token's character range without surrounding whitespace, which BPE
    tokenizers fold into the token."""
    start, end = offset
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _tag(
    spans: Iterable[Span], offsets: Sequence[tuple[int, int]], vocab: Sequence[str]
) -> list[int]:
    index = {tag: i for i, tag in enumerate(vocab)}
    tags = [index[O] if start < end else IGNORE for start, end in offsets]
    for span in spans:
        inside = [
            i for i, (s, e) in enumerate(offsets) if s < e and s < span.end and e > span.start
        ]
        for k, i in enumerate(inside):
            tags[i] = index[f"{'I' if k else 'B'}-{span.label}"]
    return tags


def _spans(offsets: Sequence[tuple[int, int]], tags: Sequence[str]) -> list[Span]:
    """Spans from BIO tags, leniently: a stray I- starts a span like a B- would."""
    spans: list[Span] = []
    current = None
    for (start, end), tag in zip(offsets, tags):
        if start >= end:
            continue  # special and padding tokens neither extend nor break a span
        if tag == O:
            current = None
            continue

        prefix, label = tag[0], tag[2:]
        if prefix == "I" and label == current:
            spans[-1] = Span(label, spans[-1].start, end)
        else:
            spans.append(Span(label, start, end))
            current = label
    return spans


def _owner(slot: Span, intents: Sequence[Span]) -> int:
    """The intent a slot overlaps most; for a slot in a gap, overlap is minus the
    distance, so the nearest intent wins."""
    return max(
        range(len(intents)),
        key=lambda k: min(intents[k].end, slot.end) - max(intents[k].start, slot.start),
    )


def _nest(intent: Span, slots: Sequence[Span], schema: Schema | None) -> IntentSpan:
    """Fit slots inside their intent: one in a gap stretches the intent over it,
    one straddling an edge is clipped to it."""
    outside = [s for s in slots if s.end <= intent.start or s.start >= intent.end]
    start = min([intent.start, *(s.start for s in outside)])
    end = max([intent.end, *(s.end for s in outside)])

    allowed = schema.intents.get(intent.label, ()) if schema else None
    fitted = tuple(
        Span(s.label, max(s.start, start), min(s.end, end))
        for s in slots
        if allowed is None or s.label in allowed
    )
    return IntentSpan(intent.label, start, end, fitted)

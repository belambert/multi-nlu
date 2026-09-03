"""Scoring predicted annotations against gold.

Generative models can drift from the input utterance, which shifts every
character offset after the drift. So the headline slot metrics compare span
*values* as multisets; `slot_span_f1` reports the stricter offset-based
agreement, and is only meaningful on faithful predictions.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Hashable, Iterable, Sequence

from multi_nlu.annotation import Annotation, is_faithful


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Scores:
    """Micro-averaged scores over a set of predictions."""

    n: int
    well_formed: float
    faithful: float
    exact_match: float
    intent: PRF
    slot: PRF
    slot_scoped: PRF
    slot_span: PRF

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        rows = [
            ("intent F1", self.intent.f1),
            ("slot F1", self.slot.f1),
            ("slot F1 (intent-scoped)", self.slot_scoped.f1),
            ("slot F1 (exact spans)", self.slot_span.f1),
            ("exact match", self.exact_match),
            ("well formed", self.well_formed),
            ("faithful", self.faithful),
        ]
        width = max(len(name) for name, _ in rows)
        lines = [f"{name:<{width}}  {value:6.2%}" for name, value in rows]
        return "\n".join([f"n = {self.n}", *lines])


def score(
    golds: Sequence[Annotation], preds: Sequence[Annotation], utterances: Sequence[str]
) -> Scores:
    """Score predictions against gold annotations of the same utterances."""
    if not (len(golds) == len(preds) == len(utterances)):
        raise ValueError("golds, preds and utterances must be the same length")

    counts: dict[str, Counter] = {key: Counter() for key in _EXTRACTORS}
    well_formed = faithful = exact = 0

    for gold, pred, utterance in zip(golds, preds, utterances):
        well_formed += not pred.malformed
        faithful += is_faithful(pred, utterance)
        exact += _structure(gold) == _structure(pred)
        for key, extract in _EXTRACTORS.items():
            counts[key] += _tally(extract(gold), extract(pred))

    n = len(golds) or 1
    return Scores(
        n=len(golds),
        well_formed=well_formed / n,
        faithful=faithful / n,
        exact_match=exact / n,
        **{key: _prf(c) for key, c in counts.items()},
    )


def _tally(gold: Iterable[Hashable], pred: Iterable[Hashable]) -> Counter:
    """Match two bags of items, counting true positives and each side's total."""
    gold_counts, pred_counts = Counter(gold), Counter(pred)
    matched = sum((gold_counts & pred_counts).values())
    return Counter(tp=matched, gold=sum(gold_counts.values()), pred=sum(pred_counts.values()))


def _prf(counts: Counter) -> PRF:
    precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
    recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
    denom = precision + recall
    return PRF(precision, recall, 2 * precision * recall / denom if denom else 0.0)


def _value(text: str) -> str:
    return " ".join(text.split()).lower()


def _intents(a: Annotation) -> list:
    return [i.label for i in a.intents]


def _slots(a: Annotation) -> list:
    return [(s.label, _value(s.text(a.text))) for i in a.intents for s in i.slots]


def _slots_scoped(a: Annotation) -> list:
    return [(i.label, s.label, _value(s.text(a.text))) for i in a.intents for s in i.slots]


def _slot_spans(a: Annotation) -> list:
    return [(i.label, *s.key) for i in a.intents for s in i.slots]


def _structure(a: Annotation) -> list:
    """The annotation as ordered content, ignoring the text between spans."""
    if a.malformed:
        return []
    return [(i.label, tuple((s.label, _value(s.text(a.text))) for s in i.slots)) for i in a.intents]


_EXTRACTORS = {
    "intent": _intents,
    "slot": _slots,
    "slot_scoped": _slots_scoped,
    "slot_span": _slot_spans,
}

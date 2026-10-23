"""Scoring predicted annotations against gold, whatever format produced them.

Everything here works on Annotations, so numbers from different formats are
directly comparable. Every score is an error rate, so lower is better. Generative
models can drift from the input utterance, which shifts every character offset
after the drift, so slot errors compare span *values* as multisets rather than
offsets.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Hashable, Iterable, Sequence

from multi_nlu.annotation import Annotation, is_faithful, value

LABEL = 16  # width of the summary's names, wide enough for the cli's timing line


@dataclass(frozen=True)
class Errors:
    """Missed and spurious items, each as a fraction of the gold count."""

    missed: float
    spurious: float

    @property
    def rate(self) -> float:
        # a wrong label is one miss plus one spurious item, so this can pass 100%
        return self.missed + self.spurious


@dataclass(frozen=True)
class Scores:
    """Error rates over a set of predictions."""

    n: int
    intent: Errors
    slot: Errors
    sentence: float
    malformed: float
    unfaithful: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        rows = [
            ("intent errors", self.intent.rate, _split(self.intent)),
            ("slot errors", self.slot.rate, _split(self.slot)),
            ("sentence errors", self.sentence, ""),
            ("malformed", self.malformed, ""),
            ("unfaithful", self.unfaithful, ""),
        ]
        lines = [f"{name:<{LABEL}}  {rate:6.2%}{split}" for name, rate, split in rows]
        return "\n".join([f"n = {self.n}", *lines])


def score(
    golds: Sequence[Annotation], preds: Sequence[Annotation], utterances: Sequence[str]
) -> Scores:
    """Score predictions against gold annotations of the same utterances."""
    if not (len(golds) == len(preds) == len(utterances)):
        raise ValueError("golds, preds and utterances must be the same length")

    counts: dict[str, Counter] = {key: Counter() for key in _EXTRACTORS}
    malformed = unfaithful = wrong = 0

    for gold, pred, utterance in zip(golds, preds, utterances):
        malformed += pred.malformed
        unfaithful += not is_faithful(pred, utterance)
        wrong += structure(gold) != structure(pred)
        for key, extract in _EXTRACTORS.items():
            counts[key] += _tally(extract(gold), extract(pred))

    n = len(golds) or 1
    return Scores(
        n=len(golds),
        sentence=wrong / n,
        malformed=malformed / n,
        unfaithful=unfaithful / n,
        **{key: _errors(c) for key, c in counts.items()},
    )


def _split(errors: Errors) -> str:
    return f"  ({errors.missed:.2%} missed, {errors.spurious:.2%} spurious)"


def _tally(gold: Iterable[Hashable], pred: Iterable[Hashable]) -> Counter:
    """Match two bags of items, counting true positives and each side's total."""
    gold_counts, pred_counts = Counter(gold), Counter(pred)
    matched = sum((gold_counts & pred_counts).values())
    return Counter(tp=matched, gold=sum(gold_counts.values()), pred=sum(pred_counts.values()))


def _errors(counts: Counter) -> Errors:
    gold = counts["gold"] or 1
    return Errors((counts["gold"] - counts["tp"]) / gold, (counts["pred"] - counts["tp"]) / gold)


def _intents(a: Annotation) -> list:
    return [i.label for i in a.intents]


def _slots(a: Annotation) -> list:
    return [(s.label, value(s.text(a.text))) for i in a.intents for s in i.slots]


def structure(a: Annotation) -> list:
    """The annotation as ordered content, ignoring the text between spans."""
    if a.malformed:
        return []
    return [(i.label, tuple((s.label, value(s.text(a.text))) for s in i.slots)) for i in a.intents]


_EXTRACTORS = {
    "intent": _intents,
    "slot": _slots,
}

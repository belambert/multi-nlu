"""The format-neutral annotation: intents and slots as character spans."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """A slot: a label over a character range of the utterance."""

    label: str
    start: int
    end: int

    def text(self, utterance: str) -> str:
        return utterance[self.start : self.end]

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.label, self.start, self.end)


@dataclass(frozen=True)
class IntentSpan(Span):
    """An intent covering a range of the utterance, with the slots inside it."""

    slots: tuple[Span, ...] = ()


@dataclass
class Annotation:
    """One utterance and the intent spans marked up over it.

    Every modeling format renders from and parses back to this, so predictions
    from different formats are scored by exactly the same code.
    """

    text: str
    intents: tuple[IntentSpan, ...] = ()
    malformed: bool = field(default=False, compare=False)

    @property
    def slots(self) -> tuple[Span, ...]:
        return tuple(s for i in self.intents for s in i.slots)


def is_faithful(annotation: Annotation, utterance: str) -> bool:
    """Whether the annotation reproduces the utterance, ignoring whitespace runs."""
    return not annotation.malformed and _norm(annotation.text) == _norm(utterance)


def _norm(text: str) -> str:
    return " ".join(text.split())

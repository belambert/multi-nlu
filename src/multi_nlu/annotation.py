"""Inline-XML annotations: parsing to spans, and rendering back to XML."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

BARE_AMP = re.compile(r"&(?!#?\w+;)")


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
    """One utterance and the intent spans marked up over it."""

    text: str
    intents: tuple[IntentSpan, ...] = ()
    malformed: bool = field(default=False, compare=False)

    @property
    def slots(self) -> tuple[Span, ...]:
        return tuple(s for i in self.intents for s in i.slots)

    def to_xml(self) -> str:
        out, pos = [], 0
        for intent in self.intents:
            out.append(escape(self.text[pos : intent.start]))
            out.append(f"<{intent.label}>")
            inner = intent.start
            for slot in intent.slots:
                out.append(escape(self.text[inner : slot.start]))
                out.append(f"<{slot.label}>{escape(slot.text(self.text))}</{slot.label}>")
                inner = slot.end
            out.append(escape(self.text[inner : intent.end]))
            out.append(f"</{intent.label}>")
            pos = intent.end
        out.append(escape(self.text[pos:]))
        return "".join(out)


def parse_xml(xml: str, utterance: str | None = None) -> Annotation:
    """Parse inline XML into spans; on unrecoverable syntax errors, yield an empty
    annotation flagged as malformed."""
    try:
        root = ET.fromstring(f"<doc>{BARE_AMP.sub('&amp;', xml.strip())}</doc>")
    except ET.ParseError:
        return Annotation(text=utterance if utterance is not None else "", malformed=True)

    chunks: list[str] = []
    pos = 0

    def take(text: str | None) -> None:
        nonlocal pos
        if text:
            chunks.append(text)
            pos += len(text)

    intents = []
    take(root.text)
    for element in root:
        start = pos
        take(element.text)
        slots = []
        for child in element:
            slot_start = pos
            take("".join(child.itertext()))
            slots.append(Span(child.tag, slot_start, pos))
            take(child.tail)
        intents.append(IntentSpan(element.tag, start, pos, tuple(slots)))
        take(element.tail)

    return Annotation(text="".join(chunks), intents=tuple(intents))


def is_faithful(annotation: Annotation, utterance: str) -> bool:
    """Whether the annotation reproduces the utterance, ignoring whitespace runs."""
    return not annotation.malformed and _norm(annotation.text) == _norm(utterance)


def _norm(text: str) -> str:
    return " ".join(text.split())

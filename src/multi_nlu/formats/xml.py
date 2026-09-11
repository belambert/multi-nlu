"""Inline-XML tagging: the model rewrites the utterance, adding tags as it copies."""

import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.formats.base import PREAMBLE, Format

BARE_AMP = re.compile(r"&(?!#?\w+;)")

INSTRUCTIONS = PREAMBLE + """

Rewrite the utterance as inline XML. Wrap each intent in a tag named after the \
intent, and inside it wrap the words that fill a slot in a tag named after the \
slot. Text that belongs to no intent stays outside every tag.

Rules:
- Reproduce the utterance verbatim. Copy every word, in order, changing nothing \
but the tags you add.
- Escape & as &amp; in the text.
- Use only the intents and slots listed below, and only slots listed under the \
intent that contains them.
- Output the tagged utterance and nothing else.

Intents and their slots:
{schema}"""


class XmlFormat(Format):
    """The model copies the utterance back, adding inline XML tags as it goes."""

    name = "xml"
    description = "inline XML tags added to a verbatim copy of the utterance"
    instructions = INSTRUCTIONS

    def render(self, annotation: Annotation) -> str:
        text = annotation.text
        out, pos = [], 0
        for intent in annotation.intents:
            out.append(escape(text[pos : intent.start]))
            out.append(f"<{intent.label}>")
            inner = intent.start
            for slot in intent.slots:
                out.append(escape(text[inner : slot.start]))
                out.append(f"<{slot.label}>{escape(slot.text(text))}</{slot.label}>")
                inner = slot.end
            out.append(escape(text[inner : intent.end]))
            out.append(f"</{intent.label}>")
            pos = max(intent.end, inner)  # a slot may overrun its intent in noisy gold
        out.append(escape(text[pos:]))
        return "".join(out)

    def parse(self, output: str, utterance: str = "") -> Annotation:
        """Parse inline XML into spans; unrecoverable syntax errors yield an empty
        annotation flagged as malformed."""
        try:
            root = ET.fromstring(f"<doc>{BARE_AMP.sub('&amp;', output.strip())}</doc>")
        except ET.ParseError:
            return Annotation(text=utterance, malformed=True)

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

    def clean(self, completion: str) -> str:
        text = super().clean(completion)
        start = text.find("<")
        return text[start:].strip() if start >= 0 else text

"""Offset annotation: the model emits intents and slots as character offsets."""

import json

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.formats.base import PREAMBLE, Format

INSTRUCTIONS = PREAMBLE + """

Emit a JSON array of intents. Each intent is an object with "intent" (the \
intent name), "start" and "end" (the character offsets of its span in the \
utterance), and "slots" (an array of its slots, each an object with "name", \
"start", and "end"). Offsets are 0-indexed and end-exclusive, so an intent's \
text is utterance[start:end]. A slot's offsets must fall within its intent's \
span.

Example, for "play jazz by miles davis":
[{{"intent": "PlayMusic", "start": 0, "end": 24, "slots": [{{"name": "genre", \
"start": 5, "end": 9}}, {{"name": "artist", "start": 13, "end": 24}}]}}]

Rules:
- Use only the intents and slots listed below, and only slots listed under the \
intent that contains them.
- Output the JSON array and nothing else.

Intents and their slots:
{schema}"""


class OffsetFormat(Format):
    """The model emits intents and slots as JSON objects of character offsets."""

    name = "offset"
    description = "a JSON list of intents and slots as character offsets"
    instructions = INSTRUCTIONS
    # no utterance to copy and offsets are short, so this format's output is much
    # shorter than xml's; 2x the utterance length leaves plenty of headroom
    budget_factor = 2

    def render(self, annotation: Annotation) -> str:
        return json.dumps(
            [
                {
                    "intent": intent.label,
                    "start": intent.start,
                    "end": intent.end,
                    "slots": [
                        {"name": slot.label, "start": slot.start, "end": slot.end}
                        for slot in intent.slots
                    ],
                }
                for intent in annotation.intents
            ],
            separators=(",", ":"),
        )

    def parse(self, output: str, utterance: str = "") -> Annotation:
        """Parse offset JSON into spans; any shape or decode error yields an empty
        annotation flagged as malformed. The text is always the input utterance:
        this format never asks the model to reproduce it, so parsed output is
        faithful by construction whenever it parses at all."""
        try:
            data = json.loads(output)
            if not isinstance(data, list):
                raise TypeError(f"expected a JSON array, got {data!r}")
            intents = tuple(
                IntentSpan(
                    str(i["intent"]),
                    _int(i["start"]),
                    _int(i["end"]),
                    tuple(
                        Span(str(s["name"]), _int(s["start"]), _int(s["end"])) for s in i["slots"]
                    ),
                )
                for i in data
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return Annotation(text=utterance, malformed=True)

        return Annotation(text=utterance, intents=intents)

    def clean(self, completion: str) -> str:
        text = super().clean(completion)
        start = text.find("[")
        return text[start:].strip() if start >= 0 else text


def _int(value: object) -> int:
    """Reject bools and non-ints so "true"/float offsets register as malformed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int offset, got {value!r}")
    return value

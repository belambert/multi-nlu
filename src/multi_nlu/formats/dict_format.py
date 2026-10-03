"""Structured JSON: the model states each intent's slots directly, as a dict."""

import json

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.formats.base import PREAMBLE, Format
from multi_nlu.metrics import value

INSTRUCTIONS = PREAMBLE + """

Output a JSON list with one object per intent found in the utterance. Each \
object has a single key, the intent's name, mapping to an object of its slots: \
each slot name mapped to the slot's text, copied verbatim from the utterance \
(the same words, not paraphrased or summarized).

Rules:
- Use only the intents and slots listed below, and only slots listed under the \
intent that contains them.
- Copy slot text exactly as it appears in the utterance.
- Output the JSON array and nothing else.

Intents and their slots:
{schema}"""


class DictFormat(Format):
    """The model states each intent's slots directly, as a JSON dict, no offsets."""

    name = "dict"
    description = "a JSON list of {intent: {slot: text}} objects, no offsets"
    instructions = INSTRUCTIONS
    # shorter than xml's verbatim copy-plus-tags: no echoed non-slot text, no offsets
    budget_factor = 3

    def render(self, annotation: Annotation) -> str:
        text = annotation.text
        objs = []
        for intent in annotation.intents:
            # a plain dict can't hold one slot label twice, so if an intent has two
            # slots sharing a name, the later one silently overwrites the earlier one
            # here. this is a real, measurable lossiness of the format, not a bug.
            slots = {slot.label: slot.text(text) for slot in intent.slots}
            objs.append({intent.label: slots})
        return json.dumps(objs, separators=(",", ":"))

    def parse(self, output: str, utterance: str = "") -> Annotation:
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return Annotation(text=utterance, malformed=True)

        if not isinstance(data, list):
            return Annotation(text=utterance, malformed=True)
        for obj in data:
            if not isinstance(obj, dict) or len(obj) != 1:
                return Annotation(text=utterance, malformed=True)
            slots = next(iter(obj.values()))
            if not isinstance(slots, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in slots.items()
            ):
                return Annotation(text=utterance, malformed=True)

        cursor = 0
        intents = []
        for obj in data:
            label, slots = next(iter(obj.items()))
            spans = []
            for slot_label, text in slots.items():
                found = _find(utterance, text, cursor)
                if found is None:
                    # a hallucinated or mismatched value has no offset to anchor it
                    # to, so it's unscoreable; drop it rather than invent a span.
                    continue
                start, end = found
                spans.append(Span(slot_label, start, end))
                cursor = end
            if spans:
                start = min(s.start for s in spans)
                end = max(s.end for s in spans)
            else:
                start, end = 0, len(utterance)
            intents.append(IntentSpan(label, start, end, tuple(spans)))

        return Annotation(text=utterance, intents=tuple(intents))

    def clean(self, completion: str) -> str:
        text = super().clean(completion)
        start = text.find("[")
        return text[start:].strip() if start >= 0 else text


def _find(utterance: str, target: str, start: int) -> tuple[int, int] | None:
    """Locate target in utterance at or after start; case-sensitive then insensitive.

    The insensitive retry scans same-length windows, so it catches case differences
    but not whitespace reflowing within the span (a narrower net than `value`'s full
    normalization allows, which is fine since slot text is otherwise copied verbatim).
    """
    idx = utterance.find(target, start)
    if idx >= 0:
        return idx, idx + len(target)

    norm_target = value(target)
    width = len(target)
    for i in range(start, len(utterance) - width + 1):
        if value(utterance[i : i + width]) == norm_target:
            return i, i + width
    return None

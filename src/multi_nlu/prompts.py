"""Chat prompts shared by few-shot prompting and fine-tuning.

Fine-tuned models see the same message layout with zero demonstrations, so a
checkpoint can be evaluated through exactly the same path as a base model.
"""

from typing import Sequence

from multi_nlu.data import Example
from multi_nlu.schema import Schema

INSTRUCTIONS = """\
You tag user utterances for a natural language understanding system.

Rewrite the utterance as inline XML. Wrap each intent in a tag named after the \
intent, and inside it wrap the words that fill a slot in a tag named after the \
slot. An utterance may express several intents; tag each one. Text that belongs \
to no intent, such as the words joining two requests, stays outside every tag.

Rules:
- Reproduce the utterance verbatim. Copy every word, in order, changing nothing \
but the tags you add.
- Escape & as &amp; in the text.
- Use only the intents and slots listed below, and only slots listed under the \
intent that contains them.
- Output the tagged utterance and nothing else.

Intents and their slots:
{schema}"""


def build_messages(
    text: str, schema: Schema, shots: Sequence[Example] = ()
) -> list[dict[str, str]]:
    """The chat messages for tagging one utterance."""
    messages = [{"role": "system", "content": system_prompt(schema)}]
    for shot in shots:
        messages.append({"role": "user", "content": shot.text})
        messages.append({"role": "assistant", "content": shot.xml})
    messages.append({"role": "user", "content": text})
    return messages


def system_prompt(schema: Schema) -> str:
    lines = [
        f"- {intent}: {', '.join(slots) if slots else '(no slots)'}"
        for intent, slots in schema.intents.items()
    ]
    return INSTRUCTIONS.format(schema="\n".join(lines))

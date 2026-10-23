"""Chat prompts shared by few-shot prompting and fine-tuning.

The chosen format supplies the instructions and renders the demonstrations, so
every approach goes through this one message layout. Fine-tuned models see it
with zero demonstrations, letting a checkpoint be evaluated exactly like a base
model.
"""

from typing import Sequence

from multi_nlu.data import Example
from multi_nlu.formats import Format
from multi_nlu.schema import Schema


def build_messages(
    text: str, schema: Schema, fmt: Format, shots: Sequence[Example] = ()
) -> list[dict[str, str]]:
    """The chat messages for annotating one utterance in the given format."""
    messages = [{"role": "system", "content": fmt.system_prompt(schema)}]
    for shot in shots:
        messages.append({"role": "user", "content": shot.text})
        messages.append({"role": "assistant", "content": fmt.render(shot.annotation)})
    messages.append({"role": "user", "content": text})
    return messages

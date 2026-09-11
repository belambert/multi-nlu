"""The Format interface: how one modeling approach writes an annotation down.

A format owns both halves of the contract — the instructions that tell the model
what to emit, and the parser that turns a generation back into an Annotation —
so adding an approach means adding a Format, and nothing else.
"""

from abc import ABC, abstractmethod

from multi_nlu.annotation import Annotation
from multi_nlu.schema import Schema

PREAMBLE = """\
You label user utterances for a natural language understanding system.

An utterance may express several intents. Each intent covers a span of the \
utterance, and the slots filling that intent sit inside its span. Words that \
belong to no intent, such as those joining two requests, are left unlabeled."""


class Format(ABC):
    """One way of writing an annotation as text for a generative model."""

    name: str
    description: str
    instructions: str  # a template with a {schema} placeholder
    budget_factor: int = 4  # generation length relative to the tokenized utterance

    @abstractmethod
    def render(self, annotation: Annotation) -> str:
        """The target text a model should produce for this annotation."""

    @abstractmethod
    def parse(self, output: str, utterance: str) -> Annotation:
        """Read a generation back; flag malformed output rather than raising."""

    def system_prompt(self, schema: Schema) -> str:
        return self.instructions.format(schema=schema_block(schema))

    def clean(self, completion: str) -> str:
        """Strip the code fences that chat models like to wrap output in."""
        text = completion.strip()
        if "```" in text:
            text = text.split("```")[1]
            for tag in ("xml", "json", "html"):
                text = text.removeprefix(tag)
        return text.strip()


def schema_block(schema: Schema) -> str:
    """The intent and slot inventory, as the prompt lists it."""
    return "\n".join(
        f"- {intent}: {', '.join(slots) if slots else '(no slots)'}"
        for intent, slots in schema.intents.items()
    )

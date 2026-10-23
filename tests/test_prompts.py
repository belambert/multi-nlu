from multi_nlu.annotation import Annotation, IntentSpan
from multi_nlu.data import Example
from multi_nlu.formats import get_format
from multi_nlu.generative.prompts import build_messages
from multi_nlu.schema import load_schema

xml = get_format("xml")
SHOT = Example("play jazz", Annotation("play jazz", (IntentSpan("PlayMusic", 0, 9, ()),)))


def test_messages_interleave_shots_and_end_on_the_query():
    messages = build_messages("book a table", load_schema("mixsnips"), xml, [SHOT])

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == xml.render(SHOT.annotation)
    assert messages[-1]["content"] == "book a table"

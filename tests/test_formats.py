import pytest

from multi_nlu.formats import DEFAULT, format_names, get_format
from multi_nlu.schema import load_schema


def test_xml_is_the_default_format():
    assert "xml" in format_names()
    assert get_format(DEFAULT).name == "xml"


def test_unknown_format_lists_the_known_ones():
    with pytest.raises(ValueError, match="xml"):
        get_format("yaml")


def test_prompt_lists_every_intent_with_its_slots():
    prompt = get_format("xml").system_prompt(load_schema("mixsnips"))

    assert "- SearchCreativeWork: object_name, object_type" in prompt
    assert all(f"- {intent}:" in prompt for intent in load_schema("mixsnips").intents)

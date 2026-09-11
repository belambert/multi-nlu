import pytest

from multi_nlu.annotation import Annotation, is_faithful
from multi_nlu.data import to_annotation
from multi_nlu.formats import get_format

TEXT = "book a table in ames and play jazz"
XML = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)
ROW = [
    {
        "intent": "BookRestaurant",
        "start": 0,
        "end": 20,
        "slots": [{"name": "city", "start": 16, "end": 20}],
    },
    {
        "intent": "PlayMusic",
        "start": 25,
        "end": 34,
        "slots": [{"name": "genre", "start": 30, "end": 34}],
    },
]

xml = get_format("xml")


def test_row_offsets_become_spans():
    a = to_annotation(TEXT, ROW)

    assert [i.label for i in a.intents] == ["BookRestaurant", "PlayMusic"]
    assert [(s.label, s.text(TEXT)) for s in a.slots] == [("city", "ames"), ("genre", "jazz")]
    assert a.intents[1].text(a.text) == "play jazz"


def test_renders_dataset_offsets_as_xml():
    assert xml.render(to_annotation(TEXT, ROW)) == XML


def test_parses_intents_and_slots_with_offsets():
    a = xml.parse(XML)

    assert a.text == TEXT
    assert a == to_annotation(TEXT, ROW)


def test_round_trips_to_xml():
    assert xml.render(xml.parse(XML)) == XML


def test_unescapes_and_reescapes_entities():
    a = xml.parse("<PlayMusic>play <artist>hall &amp; oates</artist></PlayMusic>")

    assert a.text == "play hall & oates"
    assert xml.render(a) == "<PlayMusic>play <artist>hall &amp; oates</artist></PlayMusic>"


def test_repairs_bare_ampersand():
    assert xml.parse("<PlayMusic>hall & oates</PlayMusic>").text == "hall & oates"


def test_malformed_output_is_flagged_not_raised():
    a = xml.parse("<PlayMusic>play jazz", utterance="play jazz")

    assert a.malformed and a.intents == ()
    assert not is_faithful(a, "play jazz")


def test_clean_strips_fences_and_preamble():
    assert xml.clean("Sure!\n```xml\n<PlayMusic>play jazz</PlayMusic>\n```") == (
        "<PlayMusic>play jazz</PlayMusic>"
    )


@pytest.mark.parametrize(
    "annotated,faithful",
    [
        (XML, True),
        (XML.replace("in <city>ames", "in  <city>ames"), True),  # whitespace only
        (XML.replace("ames", "boston"), False),
    ],
)
def test_faithfulness_ignores_whitespace_but_not_words(annotated, faithful):
    assert is_faithful(xml.parse(annotated), TEXT) is faithful


def test_empty_annotation_is_just_text():
    assert xml.parse("hello there") == Annotation("hello there")

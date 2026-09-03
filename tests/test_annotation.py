import pytest

from multi_nlu.annotation import Annotation, is_faithful, parse_xml

TEXT = "book a table in ames and play jazz"
XML = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)


def test_parses_intents_and_slots_with_offsets():
    a = parse_xml(XML)

    assert a.text == TEXT
    assert [i.label for i in a.intents] == ["BookRestaurant", "PlayMusic"]
    assert [(s.label, s.text(a.text)) for s in a.slots] == [("city", "ames"), ("genre", "jazz")]
    assert a.intents[1].text(a.text) == "play jazz"


def test_round_trips_to_xml():
    assert parse_xml(XML).to_xml() == XML


def test_unescapes_and_reescapes_entities():
    a = parse_xml("<PlayMusic>play <artist>hall &amp; oates</artist></PlayMusic>")

    assert a.text == "play hall & oates"
    assert a.to_xml() == "<PlayMusic>play <artist>hall &amp; oates</artist></PlayMusic>"


def test_repairs_bare_ampersand():
    assert parse_xml("<PlayMusic>hall & oates</PlayMusic>").text == "hall & oates"


def test_malformed_output_is_flagged_not_raised():
    a = parse_xml("<PlayMusic>play jazz", utterance="play jazz")

    assert a.malformed and a.intents == ()
    assert not is_faithful(a, "play jazz")


@pytest.mark.parametrize(
    "xml,faithful",
    [
        (XML, True),
        (XML.replace("in <city>ames", "in  <city>ames"), True),  # whitespace only
        (XML.replace("ames", "boston"), False),
    ],
)
def test_faithfulness_ignores_whitespace_but_not_words(xml, faithful):
    assert is_faithful(parse_xml(xml), TEXT) is faithful


def test_empty_annotation_is_just_text():
    assert parse_xml("hello there") == Annotation("hello there")

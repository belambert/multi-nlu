from multi_nlu.annotation import Annotation, IntentSpan, Span, is_faithful
from multi_nlu.formats import get_format

TEXT = "book a table in ames and play jazz"
XML = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)
GOLD = Annotation(
    TEXT,
    (
        IntentSpan("BookRestaurant", 0, 20, (Span("city", 16, 20),)),
        IntentSpan("PlayMusic", 25, 34, (Span("genre", 30, 34),)),
    ),
)

xml = get_format("xml")


def test_renders_dataset_offsets_as_xml():
    assert xml.render(GOLD) == XML


def test_parses_intents_and_slots_with_offsets():
    a = xml.parse(XML)

    assert a.text == TEXT
    assert a == GOLD


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


def test_empty_annotation_is_just_text():
    assert xml.parse("hello there") == Annotation("hello there")

import re

import pytest

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.data import load_examples
from multi_nlu.formats import get_format
from multi_nlu.metrics import structure
from multi_nlu.schema import Schema, load_schema
from multi_nlu.tagging import IGNORE, Labels, decode, encode

xml = get_format("xml")

GOLD = xml.parse(
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)
TEXT = GOLD.text
SCHEMA = Schema("toy", {"BookRestaurant": ("city",), "PlayMusic": ("genre",)})
LABELS = Labels.from_schema(SCHEMA)
SPECIAL = (0, 0)


def words(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]


def space_prefixed(text: str) -> list[tuple[int, int]]:
    """Offsets the way Qwen-style BPE reports them, leading space included."""
    return [(s - 1 if s else s, e) for s, e in words(text)]


def subwords(text: str) -> list[tuple[int, int]]:
    """Each word split in two, as a WordPiece tokenizer might."""
    out = []
    for s, e in words(text):
        mid = s + (e - s + 1) // 2
        out += [(s, mid), (mid, e)] if e - s > 1 else [(s, e)]
    return out


def tag_ids(vocab: tuple[str, ...], tags: str) -> list[int]:
    return [vocab.index(t) for t in tags.split()]


@pytest.mark.parametrize("tokenize", [words, space_prefixed, subwords])
def test_round_trip(tokenize):
    offsets = [SPECIAL, *tokenize(TEXT), SPECIAL]
    intent_ids, slot_ids = encode(GOLD, offsets, LABELS)

    assert intent_ids[0] == intent_ids[-1] == IGNORE
    assert decode(TEXT, offsets, intent_ids, slot_ids, LABELS) == GOLD


def test_subword_continuations_are_inside_tags():
    intent_ids, slot_ids = encode(GOLD, subwords(TEXT), LABELS)

    assert [LABELS.slots[i] for i in slot_ids[-2:]] == ["B-genre", "I-genre"]
    assert LABELS.intents[intent_ids[0]] == "B-BookRestaurant"


def test_round_trip_on_real_data():
    examples = load_examples(load_schema("mixsnips"), "validation", limit=500)
    labels = Labels.from_schema(load_schema("mixsnips"))

    for e in examples:
        offsets = space_prefixed(e.text)
        decoded = decode(e.text, offsets, *encode(e.annotation, offsets, labels), labels)
        assert structure(decoded) == structure(e.annotation), e.text


def test_stray_inside_tag_starts_a_span():
    text = "play jazz"
    decoded = decode(
        text,
        words(text),
        tag_ids(LABELS.intents, "I-PlayMusic I-PlayMusic"),
        tag_ids(LABELS.slots, "O I-genre"),
        LABELS,
    )

    assert decoded.intents == (IntentSpan("PlayMusic", 0, 9, (Span("genre", 5, 9),)),)


def test_same_label_spans_split_on_begin_tag():
    text = "play jazz play rock"
    decoded = decode(
        text,
        words(text),
        tag_ids(LABELS.intents, "B-PlayMusic I-PlayMusic B-PlayMusic I-PlayMusic"),
        tag_ids(LABELS.slots, "O B-genre O B-genre"),
        LABELS,
    )

    assert [(i.start, i.end) for i in decoded.intents] == [(0, 9), (10, 19)]
    assert [len(i.slots) for i in decoded.intents] == [1, 1]


def test_slot_in_a_gap_stretches_the_nearest_intent():
    text = "play some jazz"
    decoded = decode(
        text,
        words(text),
        tag_ids(LABELS.intents, "B-PlayMusic O O"),
        tag_ids(LABELS.slots, "O O B-genre"),
        LABELS,
    )

    assert decoded.intents == (IntentSpan("PlayMusic", 0, 14, (Span("genre", 10, 14),)),)


def test_slot_straddling_an_intent_edge_is_clipped():
    text = "eat in ames and play jazz"
    decoded = decode(
        text,
        words(text),
        tag_ids(LABELS.intents, "B-BookRestaurant I-BookRestaurant I-BookRestaurant O O O"),
        tag_ids(LABELS.slots, "O O B-city I-city O O"),
        LABELS,
    )

    assert decoded.intents[0].slots == (Span("city", 7, 11),)
    assert decoded.intents[0].end == 11


def test_schema_drops_slots_their_intent_does_not_take():
    text = "play ames"
    args = (
        text,
        words(text),
        tag_ids(LABELS.intents, "B-PlayMusic I-PlayMusic"),
        tag_ids(LABELS.slots, "O B-city"),
        LABELS,
    )

    assert decode(*args).intents[0].slots == (Span("city", 5, 9),)
    assert decode(*args, schema=SCHEMA).intents[0].slots == ()


def test_no_intents_decodes_to_empty_annotation():
    text = "hello there"
    decoded = decode(text, words(text), [0, 0], tag_ids(LABELS.slots, "O B-city"), LABELS)

    assert decoded == Annotation(text)


def test_labels_survive_json():
    assert Labels.from_json(LABELS.to_json()) == LABELS

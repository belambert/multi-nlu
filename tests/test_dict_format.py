import json

from multi_nlu.annotation import Annotation, IntentSpan, Span, is_faithful
from multi_nlu.data import load_examples
from multi_nlu.formats import get_format
from multi_nlu.metrics import structure
from multi_nlu.schema import load_schema

TEXT = "book a table in ames and play jazz"
# intent spans here are already exactly their slots' (min start, max end), since
# that's what parse() reconstructs - see test_intent_span_widens_to_located_slots
# below for a case where the gold span is wider than its slots.
ANNOTATION = Annotation(
    TEXT,
    intents=(
        IntentSpan("BookRestaurant", 16, 20, (Span("city", 16, 20),)),
        IntentSpan("PlayMusic", 30, 34, (Span("genre", 30, 34),)),
    ),
)

dict_fmt = get_format("dict")


def test_renders_as_compact_json_dict():
    rendered = dict_fmt.render(ANNOTATION)

    assert rendered == '[{"BookRestaurant":{"city":"ames"}},{"PlayMusic":{"genre":"jazz"}}]'


def test_round_trip_recovers_intents_and_slots():
    parsed = dict_fmt.parse(dict_fmt.render(ANNOTATION), TEXT)

    assert parsed == ANNOTATION


def test_intent_span_widens_to_located_slots():
    # this format carries no offsets for the intent itself, so parse() has to
    # rebuild it from its slots - any gold text outside the slots (e.g. "book a
    # table in" before "ames") is necessarily lost, a documented approximation.
    wide = Annotation(TEXT, intents=(IntentSpan("BookRestaurant", 0, 20, (Span("city", 16, 20),)),))

    parsed = dict_fmt.parse(dict_fmt.render(wide), TEXT)

    assert parsed.intents[0].start == 16 and parsed.intents[0].end == 20


def test_repeated_slot_label_collapses_in_render():
    # two same-named slots can't both survive a plain dict's keys; this is the
    # documented lossiness of the format, not a bug to work around.
    text = "goiano bar and taverna ok"
    first = text.index("goiano bar")
    second = text.index("taverna ok")
    annotation = Annotation(
        text,
        intents=(
            IntentSpan(
                "BookRestaurant",
                first,
                second + len("taverna ok"),
                (
                    Span("restaurant_type", first, first + len("goiano bar")),
                    Span("restaurant_type", second, second + len("taverna ok")),
                ),
            ),
        ),
    )

    rendered = dict_fmt.render(annotation)

    assert json.loads(rendered) == [{"BookRestaurant": {"restaurant_type": "taverna ok"}}]


def test_duplicate_values_resolve_to_distinct_occurrences():
    text = "play jazz then play jazz again"
    annotation = Annotation(
        text,
        intents=(
            IntentSpan("PlayMusic", 5, 9, (Span("genre", 5, 9),)),
            IntentSpan("PlayMusic", 20, 24, (Span("genre", 20, 24),)),
        ),
    )

    parsed = dict_fmt.parse(dict_fmt.render(annotation), text)

    # a forward-only cursor must land the second "jazz" on its own occurrence,
    # not re-match the first one.
    assert [s.start for i in parsed.intents for s in i.slots] == [5, 20]
    assert parsed == annotation


def test_out_of_order_intents_still_locate_earlier_slots():
    # the JSON can list intents in any order, but a slot's value may sit earlier
    # in the utterance than one already matched for a later-listed intent - a
    # single forward-only cursor would wrongly fail to find it.
    text = "play jazz then book a table in ames"
    output = '[{"BookRestaurant":{"city":"ames"}},{"PlayMusic":{"genre":"jazz"}}]'

    parsed = dict_fmt.parse(output, text)

    city = next(s for i in parsed.intents for s in i.slots if s.label == "city")
    genre = next(s for i in parsed.intents for s in i.slots if s.label == "genre")
    assert text[city.start : city.end] == "ames"
    assert text[genre.start : genre.end] == "jazz"


def test_unmatchable_value_is_dropped_not_raised():
    output = '[{"PlayMusic":{"genre":"jazz","artist":"nobody real"}}]'

    parsed = dict_fmt.parse(output, "play some jazz please")

    assert len(parsed.intents) == 1
    assert [s.label for s in parsed.intents[0].slots] == ["genre"]
    assert not parsed.malformed


def test_intent_with_no_located_slots_spans_whole_utterance():
    output = '[{"PlayMusic":{"artist":"nobody real"}}]'

    parsed = dict_fmt.parse(output, "play something")

    assert parsed.intents[0].start == 0
    assert parsed.intents[0].end == len("play something")
    assert parsed.intents[0].slots == ()


def test_invalid_json_is_malformed():
    parsed = dict_fmt.parse("not json at all", "play jazz")

    assert parsed.malformed and parsed.intents == ()


def test_wrong_shape_is_malformed():
    assert dict_fmt.parse('{"PlayMusic":{"genre":"jazz"}}', "play jazz").malformed
    assert dict_fmt.parse('["PlayMusic"]', "play jazz").malformed
    assert dict_fmt.parse('[{"PlayMusic":{"genre":1}}]', "play jazz").malformed
    assert dict_fmt.parse('[{"a":1,"b":2}]', "play jazz").malformed


def test_clean_strips_fences_and_preamble():
    completion = 'Sure!\n```json\n[{"PlayMusic":{"genre":"jazz"}}]\n```'

    assert dict_fmt.clean(completion) == '[{"PlayMusic":{"genre":"jazz"}}]'


def test_is_faithful_whenever_not_malformed():
    # text is always the input utterance by construction, so there's no drift
    # to detect for this format - faithfulness tracks malformed-ness exactly.
    parsed = dict_fmt.parse(dict_fmt.render(ANNOTATION), TEXT)
    assert is_faithful(parsed, TEXT)

    garbled = dict_fmt.parse("nonsense", TEXT)
    assert not is_faithful(garbled, TEXT)


def test_matches_real_data_most_of_the_time():
    """Render/parse round-trips most mixsnips training examples exactly; the
    rest are expected divergence from repeated-slot-label collisions."""
    examples = load_examples(load_schema("mixsnips"), "train", limit=300)

    matches = sum(
        structure(dict_fmt.parse(dict_fmt.render(e.annotation), e.text)) == structure(e.annotation)
        for e in examples
    )

    assert matches / len(examples) >= 0.95

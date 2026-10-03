import json

from multi_nlu.annotation import Annotation, is_faithful
from multi_nlu.data import load_examples, to_annotation
from multi_nlu.formats import get_format
from multi_nlu.schema import load_schema

TEXT = "book a table in ames and play jazz"
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
JSON = (
    '[{"intent":"BookRestaurant","start":0,"end":20,"slots":'
    '[{"name":"city","start":16,"end":20}]},'
    '{"intent":"PlayMusic","start":25,"end":34,"slots":'
    '[{"name":"genre","start":30,"end":34}]}]'
)

offset = get_format("offset")


def test_renders_dataset_offsets_as_json():
    assert offset.render(to_annotation(TEXT, ROW)) == JSON


def test_parses_offsets_back_to_the_annotation():
    a = offset.parse(JSON, TEXT)

    assert a.text == TEXT
    assert a == to_annotation(TEXT, ROW)


def test_round_trips_through_render_and_parse():
    gold = to_annotation(TEXT, ROW)
    assert offset.parse(offset.render(gold), TEXT) == gold


def test_malformed_json_is_flagged_not_raised():
    a = offset.parse("not json at all", TEXT)

    assert a.malformed and a.intents == ()
    assert not is_faithful(a, TEXT)


def test_wrong_shape_is_flagged_malformed():
    bad_shapes = [
        "{}",  # not a list
        '[{"intent": "PlayMusic"}]',  # missing start/end/slots
        '[{"intent": "PlayMusic", "start": "0", "end": 9, "slots": []}]',  # str offset
        '[{"intent": "PlayMusic", "start": 0, "end": 9, "slots": [{"name": "x"}]}]',
        "[1, 2, 3]",  # list of non-objects
    ]
    for output in bad_shapes:
        a = offset.parse(output, TEXT)
        assert a.malformed, output


def test_clean_strips_fences_and_preamble():
    completion = f"Sure, here you go:\n```json\n{JSON}\n```"
    assert offset.clean(completion) == JSON


def test_clean_drops_preamble_without_fences():
    assert offset.clean(f"Here is the annotation: {JSON}") == JSON


def test_parsed_output_is_faithful_by_construction():
    # text is always the utterance the model was given, never copied back, so a
    # successful parse is faithful even if the model invents nonsense offsets.
    a = offset.parse('[{"intent":"PlayMusic","start":999,"end":9999,"slots":[]}]', TEXT)

    assert not a.malformed
    assert is_faithful(a, TEXT)


def test_empty_annotation_is_just_text():
    assert offset.parse("[]", "hello there") == Annotation("hello there")


def test_round_trips_over_real_examples():
    schema = load_schema("mixsnips")
    examples = load_examples(schema, split="train", limit=200)

    for example in examples:
        rendered = offset.render(example.annotation)
        json.loads(rendered)  # sanity: render always produces valid JSON
        assert offset.parse(rendered, example.text) == example.annotation

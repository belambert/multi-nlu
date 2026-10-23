from multi_nlu.annotation import Annotation, IntentSpan
from multi_nlu.data import Example, sample_shots, to_annotation

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


def test_row_offsets_become_spans():
    a = to_annotation(TEXT, ROW)

    assert [i.label for i in a.intents] == ["BookRestaurant", "PlayMusic"]
    assert [(s.label, s.text(TEXT)) for s in a.slots] == [("city", "ames"), ("genre", "jazz")]
    assert a.intents[1].text(a.text) == "play jazz"


def test_sample_shots_spreads_over_intent_counts():
    single = Example("play jazz", Annotation("play jazz", (IntentSpan("PlayMusic", 0, 9),)))
    double = Example(
        "play jazz and go",
        Annotation(
            "play jazz and go",
            (IntentSpan("PlayMusic", 0, 9), IntentSpan("GetWeather", 14, 16)),
        ),
    )
    shots = sample_shots([single] * 5 + [double] * 5, k=4)

    assert len(shots) == 4
    assert sum(len(s.annotation.intents) == 2 for s in shots) == 2

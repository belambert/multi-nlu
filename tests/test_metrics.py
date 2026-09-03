from multi_nlu.annotation import parse_xml
from multi_nlu.metrics import score

TEXT = "book a table in ames and play jazz"
GOLD = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)


def run(pred_xml, gold_xml=GOLD, text=TEXT):
    return score([parse_xml(gold_xml)], [parse_xml(pred_xml, text)], [text])


def test_perfect_prediction():
    s = run(GOLD)

    assert s.intent.f1 == s.slot.f1 == s.slot_span.f1 == 1.0
    assert s.exact_match == s.well_formed == s.faithful == 1.0


def test_missing_intent_costs_recall_only():
    s = run("<BookRestaurant>book a table in <city>ames</city></BookRestaurant> and play jazz")

    assert s.intent.precision == 1.0
    assert s.intent.recall == 0.5
    assert s.exact_match == 0.0


def test_wrong_slot_label_scores_zero_for_that_slot():
    s = run(GOLD.replace("genre", "music_item"))

    assert s.slot.f1 == 0.5
    assert s.intent.f1 == 1.0


def test_slot_under_wrong_intent_only_hurts_scoped_score():
    swapped = (
        "<PlayMusic>book a table in <city>ames</city></PlayMusic>"
        " and <BookRestaurant>play <genre>jazz</genre></BookRestaurant>"
    )
    s = run(swapped)

    assert s.slot.f1 == 1.0
    assert s.slot_scoped.f1 == 0.0


def test_drifted_text_keeps_value_match_but_loses_span_match():
    drifted = (
        "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
        " and please <PlayMusic>play <genre>jazz</genre></PlayMusic>"
    )
    s = run(drifted)

    assert s.faithful == 0.0
    assert s.slot.f1 == 1.0
    assert s.slot_span.f1 == 0.5  # the genre span shifted by a word, the city did not


def test_malformed_prediction_scores_zero():
    s = run("<BookRestaurant>book a table in <city>ames</city>")

    assert s.well_formed == s.intent.f1 == s.slot.f1 == s.exact_match == 0.0

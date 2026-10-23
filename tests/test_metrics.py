from multi_nlu.formats import get_format
from multi_nlu.metrics import Errors, score

xml = get_format("xml")

TEXT = "book a table in ames and play jazz"
GOLD = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)


def run(pred, gold=GOLD, text=TEXT):
    return score([xml.parse(gold)], [xml.parse(pred, text)], [text])


def test_perfect_prediction():
    s = run(GOLD)

    assert s.intent.rate == s.slot.rate == 0.0
    assert s.sentence == s.malformed == s.unfaithful == 0.0


def test_missing_intent_is_a_false_negative():
    s = run("<BookRestaurant>book a table in <city>ames</city></BookRestaurant> and play jazz")

    assert s.intent == Errors(fn=0.5, fp=0.0)
    assert s.sentence == 1.0


def test_wrong_slot_label_is_one_fn_and_one_fp():
    s = run(GOLD.replace("genre", "music_item"))

    assert s.slot == Errors(fn=0.5, fp=0.5)
    assert s.slot.rate == 1.0
    assert s.intent.rate == 0.0


def test_slot_under_wrong_intent_only_breaks_exact_match():
    swapped = (
        "<PlayMusic>book a table in <city>ames</city></PlayMusic>"
        " and <BookRestaurant>play <genre>jazz</genre></BookRestaurant>"
    )
    s = run(swapped)

    assert s.slot.rate == 0.0
    assert s.sentence == 1.0


def test_drifted_text_keeps_value_match():
    drifted = (
        "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
        " and please <PlayMusic>play <genre>jazz</genre></PlayMusic>"
    )
    s = run(drifted)

    assert s.unfaithful == 1.0
    assert s.slot.rate == 0.0


def test_malformed_prediction_misses_everything():
    s = run("<BookRestaurant>book a table in <city>ames</city>")

    assert s.malformed == s.intent.fn == s.slot.fn == s.sentence == 1.0


def test_hallucinated_slots_can_push_the_rate_past_one():
    extra = "<city>ames</city> <city>x</city> <city>y</city> <city>z</city>"
    s = run(GOLD.replace("<city>ames</city>", extra))

    assert s.slot == Errors(fn=0.0, fp=1.5)
    assert s.slot.rate == 1.5


def test_summary_breaks_errors_down():
    assert "(50.00% FN, 50.00% FP)" in run(GOLD.replace("genre", "x")).summary()


def test_prediction_reads_itself_back_through_its_format():
    from multi_nlu.report import Prediction

    pred = Prediction(TEXT, GOLD, gold=GOLD, format="xml")

    assert pred.annotation == pred.gold_annotation
    assert score([pred.gold_annotation], [pred.annotation], [TEXT]).sentence == 0.0

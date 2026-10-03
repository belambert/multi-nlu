import typer
from click import unstyle

from multi_nlu.formats import get_format
from multi_nlu.report import HANG, Status, diff, render_all, summarize

LABELS = {"", "got", "want", "copied", "intents", "slots", "~ partial", "! malformed"}

xml = get_format("xml")
TEXT = "book a table in ames and play jazz"
GOLD = (
    "<BookRestaurant>book a table in <city>ames</city></BookRestaurant>"
    " and <PlayMusic>play <genre>jazz</genre></PlayMusic>"
)


def run(pred, gold=GOLD, text=TEXT):
    return diff(xml.parse(gold), xml.parse(pred, text), text, pred)


def test_perfect_prediction_is_correct_and_all_matched():
    d = run(GOLD)

    assert d.status is Status.CORRECT
    assert [m for m, _ in d.intents] == ["ok", "ok"]
    assert [m for m, _ in d.slots] == ["ok", "ok"]


def test_wrong_slot_label_is_partial_and_names_both_sides():
    d = run(GOLD.replace("genre", "music_item"))

    assert d.status is Status.PARTIAL
    assert ("missing", "PlayMusic.genre='jazz'") in d.slots
    assert ("spurious", "PlayMusic.music_item='jazz'") in d.slots


def test_missing_intent_is_marked_missing():
    d = run("<BookRestaurant>book a table in <city>ames</city></BookRestaurant> and play jazz")

    assert d.status is Status.PARTIAL
    assert ("missing", "PlayMusic") in d.intents
    assert ("ok", "BookRestaurant") in d.intents


def test_nothing_right_is_wrong():
    d = run("i have no idea")

    assert d.status is Status.WRONG
    assert all(m == "missing" for m, _ in d.intents + d.slots)
    assert not d.faithful


def test_unparseable_output_is_malformed():
    d = run("<BookRestaurant>book a table in <city>ames</city>")

    assert d.status is Status.MALFORMED


def test_slot_values_match_case_insensitively_and_display_their_own_side():
    d = run(GOLD.replace("ames", "Ames"))

    assert ("ok", "BookRestaurant.city='ames'") in d.slots  # matches shown as gold has them

    wrong_value = run(GOLD.replace("<genre>jazz", "<genre>Blues"))
    assert ("missing", "PlayMusic.genre='jazz'") in wrong_value.slots
    assert ("spurious", "PlayMusic.genre='Blues'") in wrong_value.slots


def test_render_shows_the_utterance_matched_counts_and_errors():
    out = run(GOLD.replace("genre", "music_item")).render()

    assert TEXT in out
    assert "✓1" in out  # the city slot matched
    assert "-PlayMusic.genre='jazz'" in out
    assert "+PlayMusic.music_item='jazz'" in out


def test_only_malformed_examples_show_the_raw_output():
    assert "got" not in run(GOLD).render()
    assert "got" not in run(GOLD.replace("genre", "music_item")).render()
    assert (
        "  got       <BookRestaurant>book a table" in run("<BookRestaurant>book a table").render()
    )


def test_malformed_output_shows_nothing_but_the_raw_output():
    # a malformed prediction parses to no intents or slots at all, so the
    # intents/slots rows below `got` would be nothing but every gold item
    # marked missing - noise that adds nothing since nothing was read back.
    out = run("<BookRestaurant>book a table").render()

    assert "intents" not in out
    assert "slots" not in out


def test_summarize_counts_each_status():
    diffs = [run(GOLD), run(GOLD), run("i have no idea")]

    assert summarize(diffs) == "correct 2  wrong 1"


def test_render_all_rules_off_each_example():
    out = render_all([run(GOLD), run("i have no idea")])
    rules = [line for line in out.splitlines() if set(line) == {"─"}]

    assert len(rules) == 2  # one between the examples, one closing the list
    assert out.endswith("\n")


def _diff_lines(rendered):
    return [l for l in rendered.splitlines() if l.startswith(("  want", "  copied"))]


def test_unfaithful_output_gets_a_two_line_word_diff():
    drifted = GOLD.replace(" and ", " and please ", 1)
    want, copied = _diff_lines(run(drifted).render())

    assert unstyle(want) == f"  want      {TEXT}"
    assert unstyle(copied) == f"  copied    {TEXT.replace(' and ', ' and please ', 1)}"


def test_only_the_drifting_words_are_coloured_the_shared_text_is_dimmed():
    drifted = GOLD.replace(" and ", " and please ", 1)
    want, copied = _diff_lines(run(drifted).render())

    assert typer.style("please", fg=typer.colors.RED) in copied
    assert typer.colors.GREEN not in want  # nothing is missing, only added
    assert unstyle(want) == f"  want      {TEXT}"
    assert typer.style("play jazz", dim=True) in want  # shared text is dimmed, not coloured


def test_a_faithful_prediction_shows_no_word_diff():
    assert _diff_lines(run(GOLD.replace("genre", "music_item")).render()) == []


def test_malformed_output_shows_no_word_diff():
    assert _diff_lines(run("<BookRestaurant>book a table").render()) == []


def test_each_error_gets_its_own_line_aligned_under_the_first_cell():
    out = run(GOLD.replace("genre", "music_item")).render()

    assert (
        "\n".join(
            [
                "  slots     ✓1",
                "            -PlayMusic.genre='jazz'",
                "            +PlayMusic.music_item='jazz'",
            ]
        )
        in out
    )


def test_every_line_starts_its_content_in_the_same_column():
    drifted = GOLD.replace(" and ", " and please ", 1).replace("genre", "music_item")
    lines = [unstyle(line) for line in run(drifted).render().splitlines()]

    assert len(lines) > 4  # badge, want, copied, intents, slots and its errors
    assert all(line[len(HANG)] != " " for line in lines)  # content starts in one column
    assert all(line[: len(HANG)].strip() in LABELS for line in lines)


def test_each_status_gets_its_own_colour_without_changing_the_layout():
    badges = {run(pred).render().splitlines()[0] for pred in (GOLD, "<Bad", "i have no idea")}
    colours = {line[: line.index("\x1b[0m")] for line in badges}

    assert len(colours) == 3  # correct, malformed and wrong are told apart by colour
    assert all(unstyle(line)[len(HANG)] != " " for line in badges)  # padding still right


def test_colour_is_stripped_from_the_badge_when_unstyled():
    assert unstyle(run(GOLD).render().splitlines()[0]).startswith("✓ correct   book")

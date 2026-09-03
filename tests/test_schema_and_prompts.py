import pytest

from multi_nlu.data import Example, sample_shots
from multi_nlu.prompts import build_messages, system_prompt
from multi_nlu.schema import Schema, builtin_schemas, derive_schema, load_schema

SHOT = Example("play jazz", "<PlayMusic>play <genre>jazz</genre></PlayMusic>")


def test_mixsnips_schema_loads():
    schema = load_schema("mixsnips")

    assert "mixsnips" in builtin_schemas()
    assert len(schema.intents) == 7
    assert "city" in schema.intents["BookRestaurant"]
    assert schema.dataset.path == "blambert/mixsnips-xml"


def test_loads_from_path_and_round_trips_yaml(tmp_path):
    path = tmp_path / "toy.yaml"
    path.write_text(load_schema("mixsnips").to_yaml())

    assert load_schema(path).intents == load_schema("mixsnips").intents


def test_unknown_schema_names_the_built_ins():
    with pytest.raises(FileNotFoundError, match="mixsnips"):
        load_schema("nope")


def test_derive_schema_drops_rare_slots():
    xmls = ["<PlayMusic>play <genre>jazz</genre></PlayMusic>"] * 99 + [
        "<PlayMusic>play <oops>x</oops></PlayMusic>"
    ]
    schema = derive_schema(xmls, "toy", min_freq=0.05)

    assert schema.intents == {"PlayMusic": ("genre",)}


def test_prompt_lists_every_intent_with_its_slots():
    prompt = system_prompt(load_schema("mixsnips"))

    assert "- SearchCreativeWork: object_name, object_type" in prompt
    assert all(f"- {intent}:" in prompt for intent in load_schema("mixsnips").intents)


def test_messages_interleave_shots_and_end_on_the_query():
    messages = build_messages("book a table", load_schema("mixsnips"), [SHOT])

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == SHOT.xml
    assert messages[-1]["content"] == "book a table"


def test_sample_shots_spreads_over_intent_counts():
    single = Example("play jazz", "<PlayMusic>play jazz</PlayMusic>")
    double = Example(
        "play jazz and go", "<PlayMusic>play jazz</PlayMusic> and <GetWeather>go</GetWeather>"
    )
    shots = sample_shots([single] * 5 + [double] * 5, k=4)

    assert len(shots) == 4
    assert sum(len(s.annotation.intents) == 2 for s in shots) == 2


def test_schema_slots_are_the_union():
    assert load_schema("mixsnips").slots[:2] == ("album", "artist")

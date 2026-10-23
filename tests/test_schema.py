import pytest

from multi_nlu.annotation import Annotation, IntentSpan, Span
from multi_nlu.schema import builtin_schemas, derive_schema, load_schema


def test_mixsnips_schema_loads():
    schema = load_schema("mixsnips")

    assert "mixsnips" in builtin_schemas()
    assert len(schema.intents) == 7
    assert "city" in schema.intents["BookRestaurant"]
    assert schema.dataset.path == "blambert/mixsnips-intent-spans"
    assert schema.dataset.intents_column == "intents"


def test_loads_from_path_and_round_trips_yaml(tmp_path):
    path = tmp_path / "toy.yaml"
    path.write_text(load_schema("mixsnips").to_yaml())

    assert load_schema(path).intents == load_schema("mixsnips").intents


def test_unknown_schema_names_the_built_ins():
    with pytest.raises(FileNotFoundError, match="mixsnips"):
        load_schema("nope")


def test_derive_schema_drops_rare_slots():
    jazz = Annotation("play jazz", (IntentSpan("PlayMusic", 0, 9, (Span("genre", 5, 9),)),))
    oops = Annotation("play x", (IntentSpan("PlayMusic", 0, 6, (Span("oops", 5, 6),)),))
    schema = derive_schema([jazz] * 99 + [oops], "toy", min_freq=0.05)

    assert schema.intents == {"PlayMusic": ("genre",)}


def test_schema_slots_are_the_union():
    assert load_schema("mixsnips").slots[:2] == ("album", "artist")

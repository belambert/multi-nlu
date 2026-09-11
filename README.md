# multi-nlu

Multi-intent NLU with generative language models. A single utterance can carry
several intents, each with its own slots:

    book a table in ames and play jazz

Instead of predicting intents and BIO slot tags separately, this repo treats
NLU as constrained rewriting: the model copies the utterance back and adds
inline XML tags around the intents and slots it finds.

    <BookRestaurant>book a table in <city>ames</city></BookRestaurant> and <PlayMusic>play <genre>jazz</genre></PlayMusic>

One format covers overlapping requests, nested slots and arbitrary label sets,
and it works both by prompting an open-weight model few-shot and by fine-tuning
one with LoRA. Both paths use the same prompt, the same parser and the same
metrics, so their numbers are comparable.

The default dataset is [blambert/mixsnips-xml](https://huggingface.co/datasets/blambert/mixsnips-xml)
— MixSNIPS utterances with 7 intents and 39 slots, mixed one to three intents
per utterance.

## Install

    uv sync

Models and datasets come from the Hugging Face Hub; run `hf auth login` (or set
`HF_TOKEN`) if either is private.

## Modeling options

- Treat it like XML tagging
- Output JSON offset annotations
- Output a JSON dict version e.g. `[{"PlayMusic": {"slot1": ...}}]`
- Various ways of using an encoder model?


## TODO

- [ ] In dataset repo, use offset annotations instead of XML
- [ ] In this repo, support modeling this in multiple ways. E.g. the
    model can output XML tags, or offset annotations, or could output a
    structured version directly, e.g. JSON. Investigate which works better.
- [ ] add wandb/trackio
- [ ] Do the modeling with an encoder model
- [ ] Evaluation? Does order matter?
- [ ] what else?


## Use

Prompt a base model few-shot and score it:

    uv run multi-nlu predict -m Qwen/Qwen3-1.7B --shots 8 --split test -n 200

Fine-tune a LoRA adapter, then evaluate it:

    uv run multi-nlu train -m Qwen/Qwen3-1.7B -o runs/qwen3-1.7b
    uv run multi-nlu predict -m Qwen/Qwen3-1.7B --adapter runs/qwen3-1.7b --split test

Predictions can be saved and re-scored without rerunning the model:

    uv run multi-nlu predict ... -o preds.jsonl
    uv run multi-nlu score preds.jsonl

Other commands: `show` prints annotated examples and the system prompt,
`schemas` lists the bundled schemas, `derive-schema` builds a new one. Every
command takes `--help`.

The device is chosen automatically — cuda, then mps, then cpu — and `--device`
overrides it. Training uses bf16 on cuda and fp32 elsewhere.

## Metrics

Generated text can drift from the input, which shifts every character offset
after the drift, so the headline slot scores compare span *values* rather than
positions.

| Metric                  | Counts a match when                                   |
| ----------------------- | ----------------------------------------------------- |
| intent F1               | the intent label appears as often as in gold          |
| slot F1                 | a slot label and its text match                       |
| slot F1 (intent-scoped) | that slot also sits under the right intent            |
| slot F1 (exact spans)   | the slot covers the same characters — strict          |
| exact match             | every intent and slot in the utterance is right       |
| well formed             | the output parses as XML                              |
| faithful                | the output reproduces the utterance, whitespace aside |

`well formed` and `faithful` measure whether the model respected the format at
all; they are usually the first thing fine-tuning fixes.

## Schemas

A schema is a YAML file naming the intents, the slots each intent takes, and the
dataset it came from. The bundled one lives in `src/multi_nlu/schemas/`:

    name: mixsnips
    dataset:
      path: blambert/mixsnips-xml
      text_column: text
      target_column: xml
    intents:
      GetWeather:
        - city
        - timeRange

To use a different dataset, write a schema for it — or derive one from its gold
annotations — and pass it to any command:

    uv run multi-nlu derive-schema org/my-dataset -o my-dataset.yaml
    uv run multi-nlu predict --schema my-dataset.yaml -m Qwen/Qwen3-1.7B

`derive-schema` drops slots seen in under 1% of an intent's occurrences
(`--min-freq`), which filters annotation noise. Read the result before using it.

## Layout

    src/multi_nlu/
        schema.py       schemas and the datasets they describe
        annotation.py   inline XML <-> intent and slot spans
        data.py         loading examples, sampling few-shot demonstrations
        prompts.py      the prompt shared by prompting and fine-tuning
        models.py       model loading, device and dtype selection
        predict.py      batched generation
        metrics.py      scoring
        train.py        LoRA fine-tuning
        cli.py          command line interface

## Development

    uv run pytest
    uv run black . && uv run isort . && uv run mypy src

The tests cover parsing, scoring, prompting and schema handling, and need
neither a model nor the dataset.

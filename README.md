# multi-nlu

Multi-intent NLU with generative language models. A single utterance can carry
several intents, each with its own slots:

    book a table in ames and play jazz

Instead of predicting intents and BIO slot tags separately, this repo has a
generative model write the whole annotation down in one pass — and compares
*several ways of writing it down*, since the choice of output format is itself a
modeling decision. The first is inline XML: the model copies the utterance back
and adds tags around the intents and slots it finds.

    <BookRestaurant>book a table in <city>ames</city></BookRestaurant> and <PlayMusic>play <genre>jazz</genre></PlayMusic>

Every format parses back to the same span-based `Annotation`, so one parser-
agnostic scorer grades them all and the numbers are directly comparable — across
formats, and between few-shot prompting and LoRA fine-tuning.

The default dataset is
[blambert/mixsnips-intent-spans](https://huggingface.co/datasets/blambert/mixsnips-intent-spans)
— MixSNIPS utterances with 7 intents and 39 slots, one to three intents each,
annotated as character offsets rather than in any one output format.

## Install

    uv sync

Models and datasets come from the Hugging Face Hub; run `hf auth login` (or set
`HF_TOKEN`) if either is private.

## Formats

A format owns both halves of the contract — the instructions telling the model
what to emit, and the parser reading a generation back into spans — so adding an
approach means adding one `Format` and nothing else. `--format` picks one, and
`multi-nlu formats` lists what is implemented.

| Format   | Status | The model emits                                                                         |
| -------- | ------ | ---------------------------------------------------------------------------------------- |
| `xml`    | done   | the utterance copied verbatim with inline XML tags added                                 |
| `dict`   | done   | `[{"PlayMusic": {"genre": …}}]`, slots copied verbatim; a repeated slot label collapses  |

`xml` and `dict` both exist today, and an encoder-based baseline sits outside
this interface.

## Use

Prompt a base model few-shot and score it:

    uv run multi-nlu predict -m Qwen/Qwen3-1.7B --shots 8 --split test -n 200

Fine-tune a LoRA adapter, then evaluate it:

    uv run multi-nlu train -m Qwen/Qwen3-1.7B -o runs/qwen3-1.7b
    uv run multi-nlu predict -m Qwen/Qwen3-1.7B --adapter runs/qwen3-1.7b --split test

Both commands take `--format`, and training an adapter on a format evaluates
through the same one:

    uv run multi-nlu train --format xml -o runs/xml

Training logs to Weights & Biases by default (project `multi-nlu`), tracking
the hyperparameters and loss curves; pass `--no-wandb` to disable it and
`--wandb-project` to log elsewhere.

Predictions can be saved and re-scored without rerunning the model; each row
records the format that produced it, so `score` needs no flags:

    uv run multi-nlu predict ... -o preds.jsonl
    uv run multi-nlu score preds.jsonl

Both `predict` and `score` take `--detail` (`-d`) to print the examples
themselves — `errors` for the ones that went wrong, `all` for every one:

    uv run multi-nlu score preds.jsonl -d errors

Each example shows its status and a diff of its intents and slots. Matches
collapse to a count so the errors stand out, one per line; `-` marks something
gold had and the prediction missed, `+` something the prediction invented.

Reading a `-` against the `+` under it usually names the mistake.

A `correct` example matches gold on every intent and slot, `partial` gets some
of them, `wrong` none, and `malformed` did not parse at all. Each status has its
own colour on a terminal — green, yellow, red and magenta — so a long run can be
skimmed. Only a `malformed`
example prints the generation itself, since nothing could be read out of it;
anywhere else the marks and the diff below say more than the raw output does.

A format like `xml` asks the model to copy the utterance back verbatim and only
add tags, so when it paraphrases instead, every character offset after the
change stops lining up with gold. Those examples get a two-line word diff of
the utterance against the text the model actually copied. The text the two
share is dimmed, so what stands out is the words that differ — green for what
the utterance says, red for what the prediction put there:

    want      ... in panama for two people and give the current book a zero of 6
    copied    ... in panama for two people and please give the current book a zero of 6

Colour is dropped when the output is not a terminal, so piping to a file or a
pager stays readable.

Other commands: `show` prints annotated examples in a format and the system
prompt, `formats` and `schemas` list what is bundled, `derive-schema` builds a
new schema. Every command takes `--help`.

### Comparing formats

`scripts/compare-formats.sh` runs `predict` on the same sample once per format
for each instruct Qwen3.5 model under 40B, printing each one's scores in turn:

    ./scripts/compare-formats.sh

Predictions are saved as `predictions/few-shot/<model>-<format>.jsonl`; rescore
one with `multi-nlu score`. `scripts/compare-finetuned.sh` does the same for
LoRA fine-tunes, writing `predictions/finetuned/<model>-<format>.jsonl`.

Edit the models or args in the script directly to compare something else.

The device is chosen automatically — cuda, then mps, then cpu — and `--device`
overrides it. Training uses bf16 on cuda and fp32 elsewhere.

## Metrics

Scoring happens on parsed annotations, never on raw text, so every format is
graded identically. Generated text can drift from the input utterance, which
shifts every character offset after the drift, so the headline slot scores
compare span *values* rather than positions.

| Metric                  | Counts a match when                                   |
| ----------------------- | ----------------------------------------------------- |
| intent F1               | the intent label appears as often as in gold          |
| slot F1                 | a slot label and its text match                       |
| slot F1 (intent-scoped) | that slot also sits under the right intent            |
| slot F1 (exact spans)   | the slot covers the same characters — strict          |
| exact match             | every intent and slot in the utterance is right       |
| well formed             | the output parses in its format                       |
| faithful                | the output reproduces the utterance, whitespace aside |

`well formed` and `faithful` measure whether the model respected the format at
all; they are usually the first thing fine-tuning fixes. Formats that emit
offsets rather than a copy of the utterance are faithful by construction, which
is part of what the comparison is meant to expose.

## Results

Few-shot prompting (8 shots) on the full test split (n = 2199), from
`scripts/compare-formats.sh`. Slot F1 is by value, `scoped` is intent-scoped, and
`exact` compares character spans.

### Few-Shot

| Model          | Format   | Intent F1 | Slot F1 | Slot F1 scoped | Slot F1 exact | Exact match | Well formed | Faithful |
| -------------- | -------- | --------- | ------- | -------------- | ------------- | ----------- | ----------- | -------- |
| Qwen3.5-2B     | `xml`    |    88.58% |  46.27% |         44.06% |        22.65% |       4.91% |      97.18% |   25.51% |
| Qwen3.5-2B     | `dict`   |    82.48% |  43.51% |         42.19% |        42.16% |       3.55% |      99.68% |   99.68% |
| _other models_ |          |           |         |                |               |             |             |          |

Few-shot `xml` rarely copies the utterance back faithfully (25.5%), which is
why its exact-span slot F1 is about half its by-value score; `dict` copies slot
text rather than the whole utterance and loses almost nothing.

### Fine-Tuned (LoRA)

| Model          | Format   | Intent F1 | Slot F1 | Slot F1 scoped | Slot F1 exact | Exact match | Well formed | Faithful |
| -------------- | -------- | --------- | ------- | -------------- | ------------- | ----------- | ----------- | -------- |
| _to be added_  |          |           |         |                |               |             |             |          |

## Schemas

A schema is a YAML file naming the intents, the slots each intent takes, and the
dataset it came from. The bundled one lives in `src/multi_nlu/schemas/`:

    name: mixsnips
    dataset:
      path: blambert/mixsnips-intent-spans
      text_column: text
      intents_column: intents
    intents:
      GetWeather:
        - city
        - timeRange

`intents_column` holds the offset annotations: a list of
`{intent, start, end, slots: [{name, start, end}]}`.

To use a different dataset, write a schema for it — or derive one from its gold
annotations — and pass it to any command:

    uv run multi-nlu derive-schema org/my-dataset -o my-dataset.yaml
    uv run multi-nlu predict --schema my-dataset.yaml -m Qwen/Qwen3-1.7B

`derive-schema` drops slots seen in under 1% of an intent's occurrences
(`--min-freq`), which filters annotation noise. Read the result before using it.

## TODO

- [ ] Compare both formats
- [ ] Do the modeling with an encoder model
- [x] add wandb/trackio
- [ ] Evaluation? Does order matter?
- [ ] Fix the 2 train rows whose slot span overruns its intent span (dataset repo)
- [ ] what else?

## Layout

    src/multi_nlu/
        annotation.py   the format-neutral spans everything converts to
        schema.py       schemas and the datasets they describe
        formats/        one module per way of writing an annotation down
            base.py     the Format interface and shared prompt preamble
            xml.py      inline XML tagging
            dict_format.py  structured-JSON tagging
        data.py         loading examples, sampling few-shot demonstrations
        prompts.py      the message layout shared by prompting and fine-tuning
        models.py       model loading, device and dtype selection
        predict.py      batched generation
        metrics.py      scoring, on annotations rather than text
        report.py       per-example diffs of gold against a prediction
        train.py        LoRA fine-tuning
        cli.py          command line interface

## Development

    uv run pytest
    uv run black . && uv run isort . && uv run mypy src

The tests cover the span model, the `xml` and `dict` formats,
scoring, per-example diffs, prompting and schema handling, and need neither a
model nor a GPU; one `dict` test pulls real examples from the dataset to check
round-trip fidelity, so it needs network access the first time it isn't cached.

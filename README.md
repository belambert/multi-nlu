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

`xml` and `dict` both exist today. The [token tagger](#token-tagger) baseline
sits outside this interface, since it classifies tokens instead of generating
text.

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

Edit the models or args in the script directly to compare something else.

The device is chosen automatically — cuda, then mps, then cpu — and `--device`
overrides it. Training uses bf16 on cuda and fp32 elsewhere.

### Token Tagger

The non-generative baseline is a BERT-style sequence tagger with two heads on
a shared backbone. One head tags intent spans and the other tags slots, each
with BIO labels over subword tokens:

    book    a       table   in      ames    and  play         jazz
    B-Book  I-Book  I-Book  I-Book  I-Book  O    B-PlayMusic  I-PlayMusic
    O       O       O       O       B-city  O    O            B-genre

Intents never nest, and slots sit inside intents, so these two flat layers
capture the whole annotation. Decoding puts each slot under the intent span
that contains it. A slot just outside every intent stretches the nearest one to
cover it, and a slot the schema doesn't allow under its intent is dropped. The
loss is the sum of both heads' cross-entropies; `--slot-weight` rebalances it.

    uv run multi-nlu tag-train -m answerdotai/ModernBERT-base -o runs/tagger
    uv run multi-nlu tag-predict -t runs/tagger --split test -o preds.jsonl

`tag-predict` writes the same JSONL as `predict`, rendered through `xml`, so
`score` and `--detail` work on it unchanged. A tagger is well formed and
faithful by construction.

Any backbone `AutoModel` can load works, decoders included.
`scripts/compare-taggers.sh` trains and scores ModernBERT, BERT, RoBERTa and
Qwen3.5-0.8B. Qwen tags with left context only, which is the point of the
comparison:

    ./scripts/compare-taggers.sh

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

| Model           | Format | Intent F1 | Slot F1 | Slot F1 scoped | Slot F1 exact | Exact match | Well formed | Faithful |
| --------------- | ------ | --------- | ------- | -------------- | ------------- | ----------- | ----------- | -------- |
| Qwen3.5-0.8B    | `xml`  |    77.49% |  34.04% |         31.56% |        13.53% |       2.32% |      85.45% |   15.01% |
| Qwen3.5-0.8B    | `dict` |    80.61% |  35.78% |         32.99% |        32.92% |       2.09% |      87.86% |   87.86% |
| Qwen3.5-2B      | `xml`  |    88.58% |  46.27% |         44.06% |        22.65% |       4.91% |      97.18% |   25.51% |
| Qwen3.5-2B      | `dict` |    82.48% |  43.51% |         42.19% |        42.16% |       3.55% |      99.68% |   99.68% |
| Qwen3.5-4B      | `xml`  |    88.88% |  65.14% |         63.78% |        51.07% |      15.19% |      98.41% |   68.44% |
| Qwen3.5-4B      | `dict` |    93.19% |  65.43% |         64.42% |        64.39% |       9.10% |     100.00% |  100.00% |
| Qwen3.5-9B      | `xml`  |    92.87% |  69.34% |         68.37% |        57.76% |      19.46% |      99.14% |   75.17% |
| Qwen3.5-9B      | `dict` |    93.65% |  69.04% |         68.55% |        68.51% |      12.14% |     100.00% |  100.00% |
| Qwen3.5-27B     | `xml`  |    95.68% |  80.48% |         80.11% |        76.40% |      33.42% |      99.91% |   91.72% |
| Qwen3.5-27B     | `dict` |    95.90% |  80.40% |         80.11% |        80.07% |      26.65% |     100.00% |  100.00% |
| Qwen3.5-35B-A3B | `xml`  |    90.29% |  74.83% |         73.95% |        67.65% |      25.42% |      95.45% |   82.86% |
| Qwen3.5-35B-A3B | `dict` |    94.16% |  74.76% |         74.36% |        74.35% |      19.87% |      99.23% |   99.23% |

Few-shot `xml` rarely copies the utterance back faithfully (15-92%), which is
why its exact-span slot F1 trails its by-value score; `dict` copies slot text
rather than the whole utterance and loses almost nothing. Scale helps slots
(35% at 0.8B, 80% at 27B), but exact match stays low (at most 33.4%). The
35B-A3B mixture-of-experts model trails the dense 27B and is less well formed
(95% against 99.9-100%).

### Fine-Tuned (LoRA)

| Model          | Format   | Intent F1 | Slot F1 | Slot F1 scoped | Slot F1 exact | Exact match | Well formed | Faithful |
| -------------- | -------- | --------- | ------- | -------------- | ------------- | ----------- | ----------- | -------- |
| Qwen3.5-0.8B   | `xml`    |    98.17% |  95.35% |         95.12% |        94.86% |      79.90% |     100.00% |   99.32% |
| Qwen3.5-0.8B   | `dict`   |    98.28% |  95.58% |         95.45% |        95.36% |      79.40% |     100.00% |  100.00% |
| Qwen3.5-2B     | `xml`    |    98.79% |  95.57% |         95.51% |        95.16% |      79.99% |     100.00% |   99.32% |
| Qwen3.5-2B     | `dict`   |    98.63% |  95.78% |         95.70% |        95.70% |      80.99% |     100.00% |  100.00% |
| Qwen3.5-4B     | `xml`    |    98.56% |  96.55% |         96.39% |        96.39% |      84.36% |     100.00% |  100.00% |
| Qwen3.5-4B     | `dict`   |    98.29% |  96.50% |         96.33% |        96.33% |      83.81% |     100.00% |  100.00% |
| Qwen3.5-9B     | `xml`    |    98.56% |  96.99% |         96.98% |        96.86% |      86.40% |     100.00% |   99.73% |
| Qwen3.5-9B     | `dict`   |    98.72% |  96.75% |         96.74% |        96.74% |      84.95% |     100.00% |  100.00% |
| _other models_ |          |           |         |                |               |             |             |          |

One epoch of LoRA on the 0.8B model beats few-shot 27B by about 15 points of
slot F1 and lifts exact match from 27-33% to about 80%. It also closes the
format gap: `xml` and `dict` land within 0.5 points on intent and slot F1 at
every size, and both are fully well formed. Exact match differs more (up to 1.5
points, favoring `xml` at 4B and 9B).

Scale helps only modestly once the model is fine-tuned. Going from 0.8B to 2B
adds about 0.2 points of slot F1 and up to 1.6 points of exact match; 2B to 4B
adds 0.7-1.0 points of slot F1 and 2.8-4.4 points of exact match; 4B to 9B adds
0.3-0.4 points of slot F1 and 1.1-2.0 points of exact match, reaching 86.4%
exact match with `xml`.

### Token Tagger

Three epochs of full fine-tuning with `tag-train` defaults, scored by
`tag-predict` on the full test split.

| Backbone        | Intent F1 | Slot F1 | Slot F1 scoped | Slot F1 exact | Exact match | Well formed | Faithful |
| --------------- | --------- | ------- | -------------- | ------------- | ----------- | ----------- | -------- |
| ModernBERT-base |    98.18% |  96.03% |         95.84% |        95.84% |      83.17% |     100.00% |  100.00% |
| _other models_  |           |         |                |               |             |             |          |

The 149M-parameter ModernBERT tagger beats LoRA Qwen3.5-0.8B on every score,
by about 3 points of exact match. It also tags in 2.7 ms per utterance on
Apple-silicon mps at batch size 64, since it needs one forward pass and no
generation.

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
- [x] Do the modeling with an encoder model
- [ ] Bidirectional attention for decoder taggers, to separate the weights from the mask
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
        tagging.py      spans as two layers of BIO tags, and back
        tagger.py       the two-headed token tagger: model, training, prediction
        cli.py          command line interface

## Development

    uv run pytest
    uv run black . && uv run isort . && uv run mypy src

The tests cover the span model, the `xml` and `dict` formats, BIO tagging,
scoring, per-example diffs, prompting and schema handling, and need neither a
model nor a GPU. One `dict` test and one tagging test pull real examples from
the dataset to check round-trip fidelity, so they need network access the first
time it isn't cached.

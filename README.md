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

    uv run multi-nlu predict -m Qwen/Qwen3.5-2B --shots 8 --split test -n 200

Fine-tune a LoRA adapter, then evaluate it:

    uv run multi-nlu train -m Qwen/Qwen3.5-2B -o runs/qwen3.5-2b
    uv run multi-nlu predict -m Qwen/Qwen3.5-2B --adapter runs/qwen3.5-2b --split test

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

`predict -o` also writes `preds.meta.json` beside the predictions: the model,
adapter, format, sample, batch size, device and inference time. `score` reads it
to report the time per utterance again. The time is the whole batched
generation, excluding model loading, so it only compares across runs on the
same device and batch size.

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

`scripts/compare-few-shot.sh` prompts each instruct Qwen3.5 model under 40B with
8 shots, once per format on the same sample, printing each one's scores in turn
and saving its predictions under `runs/few-shot/` for re-scoring with `score`:

    ./scripts/compare-few-shot.sh

`scripts/compare-finetuned.sh` does the same for LoRA fine-tunes, training an
adapter per model and format under `runs/` and then evaluating it:

    ./scripts/compare-finetuned.sh

Both take model ids to run only those, e.g.
`./scripts/compare-finetuned.sh Qwen/Qwen3.5-0.8B`. Both are safe to rerun: runs
with saved predictions are skipped and finished adapters are reused, so delete
a run's files to redo it. A failed run is listed at the end rather than stopping
the others. Training logs to wandb, so run `wandb login` first or set
`WANDB_MODE=offline` on a machine that isn't logged in.

The device is chosen automatically — cuda, then mps, then cpu — and `--device`
overrides it. Training uses bf16 on cuda and fp32 elsewhere.

## Metrics

Scoring happens on parsed annotations, never on raw text, so every format is
graded identically. Generated text can drift from the input utterance, which
shifts every character offset after the drift, so slot errors compare span
*values* rather than positions. Every score is an error rate, so lower is
better.

| Metric          | Counts an error when                                          |
| --------------- | ------------------------------------------------------------- |
| total errors    | any intent or slot error, pooled over both                    |
| intent errors   | a gold intent is missed, or a predicted one is spurious       |
| slot errors     | a gold slot (label and text) is missed, or one is spurious    |
| sentence errors | any intent or slot in the utterance is wrong                  |
| malformed       | the output does not parse in its format                       |
| unfaithful      | the output does not reproduce the utterance, whitespace aside |

Intent and slot errors are divided by the number of gold items and broken down
into false negatives (FN, missed) and false positives (FP, spurious), which
`--detail` marks `-` and `+`. A wrong label counts as one of each, so a model
that invents many slots can exceed 100%.

Total errors divides the intent and slot errors together by the gold intents
and slots together, so it is the one number to rank models by. Slots outnumber
intents, so it mostly tracks slot errors, and a missed intent costs its slots
too. It resembles the semantic error rate, but that aligns predictions to gold
and counts a wrong label once, as a substitution, where this counts it twice.

`malformed` and `unfaithful` measure whether the model respected the format at
all; they are usually the first thing fine-tuning fixes. Formats that emit
offsets rather than a copy of the utterance are faithful by construction, which
is part of what the comparison is meant to expose.

## Results

All results are on the full test split (n = 2199).

### Few-Shot

Few-shot prompting (8 shots), from `scripts/compare-few-shot.sh`. Every score is
an error rate (see [Metrics](#metrics)), timed as for the fine-tuned runs below.
The re-run with error rates has so far reached 4B; larger models follow.

| Model        | Format | Total      | Intent     | Slot    | Sentence   | Malformed | Unfaithful | ms / utt |
| ------------ | ------ | ---------- | ---------- | ------- | ---------- | --------- | ---------- | -------- |
| Qwen3.5-0.8B | `xml`  |     95.97% |     40.76% | 117.37% |     97.68% |    14.55% |     84.99% |      568 |
| Qwen3.5-0.8B | `dict` |     88.01% |     35.79% | 108.26% |     97.91% |    12.14% |     12.14% |      358 |
| Qwen3.5-2B   | `xml`  |     79.57% |     22.28% | 101.78% |     95.09% |     2.82% |     74.49% |      749 |
| Qwen3.5-2B   | `dict` |     77.73% |     30.80% |  95.93% |     96.45% |     0.32% |      0.32% |      457 |
| Qwen3.5-4B   | `xml`  |     54.29% |     22.26% |  66.72% | **84.81%** |     1.59% |     31.56% |     1771 |
| Qwen3.5-4B   | `dict` | **50.68%** | **13.58%** |  65.06% |     90.90% |     0.00% |      0.00% |     1058 |

Few-shot errors stay far above fine-tuned ones: slot errors exceed 100% below
4B, since the models both miss most gold slots and invent spurious ones. Size
helps most between 2B and 4B (about 25 points of total errors). `dict` has
fewer total errors at every size so far (2-8 points) and runs 40% faster, but
`xml` gets more utterances fully right at 4B (84.81% sentence errors against
90.90%). `xml` paraphrases the utterance often (32-85%), while `dict` is
unfaithful only when malformed.

### Fine-Tuned (LoRA)

One epoch of LoRA per model and format, from `scripts/compare-finetuned.sh`.
Every score is an error rate (see [Metrics](#metrics)); no run produced
malformed output. Time per utterance is batched generation (batch size 8) on a
DGX Spark.

| Model        | Format | Total     | Intent | Slot  | Sentence | Unfaithful | ms / utt |
| ------------ | ------ | --------- | ------ | ----- | -------- | ---------- | -------- |
| Qwen3.5-0.8B | `xml`  |     7.74% |  3.66% | 9.32% |   20.10% |      0.68% |      388 |
| Qwen3.5-0.8B | `dict` |     7.32% |  3.44% | 8.83% |   20.60% |      0.00% |      240 |
| Qwen3.5-2B   | `xml`  |     7.08% |  2.43% | 8.88% |   20.01% |      0.68% |      566 |
| Qwen3.5-2B   | `dict` |     6.85% |  2.74% | 8.44% |   19.01% |      0.00% |      345 |
| Qwen3.5-4B   | `xml`  |     5.77% |  2.88% | 6.90% |   15.64% |      0.00% |     1330 |
| Qwen3.5-4B   | `dict` |     5.99% |  3.42% | 6.99% |   16.19% |      0.00% |      817 |
| Qwen3.5-9B   | `xml`  | **5.14%** |  2.88% | 6.02% |   13.60% |      0.27% |     1914 |
| Qwen3.5-9B   | `dict` |     5.40% |  2.56% | 6.50% |   15.05% |      0.00% |     1167 |

Fine-tuning closes most of the gap that few-shot prompting leaves: even the
0.8B model gets four in five utterances fully right, against about one in seven
for few-shot 4B. Errors fall steadily with size, most between 2B and 4B (about
1.1 points of total errors), reaching 5.14% with 9B `xml`.

The two formats stay close at every size. `dict` is slightly more accurate at
0.8B and 2B and `xml` at 4B and 9B, by 0.2-0.4 points of total errors. `dict` is
35-40% faster, since it copies only slot text rather than the whole utterance,
and it is always faithful, while `xml` paraphrases the utterance in up to 0.7%
of examples.

Fine-tuning Qwen3.5-27B runs out of memory on the DGX Spark's 128 GB of unified
memory with the current settings, so it has no results yet.

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
    uv run multi-nlu predict --schema my-dataset.yaml -m Qwen/Qwen3.5-2B

`derive-schema` drops slots seen in under 1% of an intent's occurrences
(`--min-freq`), which filters annotation noise. Read the result before using it.

## TODO

- [ ] Re-run the few-shot experiments to get error-rate figures
- [ ] Fit 27B fine-tuning in memory (e.g. 4-bit loading or gradient checkpointing)
- [ ] Do the modeling with an encoder model
- [ ] Switch to alignment-based evaluation using an external library
- [ ] Fix the 2 train rows whose slot span overruns its intent span (dataset repo)

## Layout

    src/multi_nlu/
        annotation.py   the format-neutral spans everything converts to
        schema.py       schemas and the datasets they describe
        formats/        one module per way of writing an annotation down
            base.py     the Format interface and shared prompt preamble
            xml.py      inline XML tagging
            dict_format.py  structured-JSON tagging
        data.py         loading examples, sampling few-shot demonstrations
        device.py       device and dtype selection, timing
        metrics.py      scoring, on annotations rather than text
        report.py       saved predictions, their scores and per-example diffs
        cli.py          command line interface
        generative/     prompting and LoRA fine-tuning decoder LMs
            models.py   causal LM and tokenizer loading
            prompts.py  the message layout shared by prompting and fine-tuning
            predict.py  batched generation
            train.py    LoRA fine-tuning

## Development

    uv run pytest
    uv run black . && uv run isort . && uv run mypy src

The tests cover the span model, the `xml` and `dict` formats,
scoring, per-example diffs, prompting and schema handling, and need neither a
model nor a GPU; one `dict` test pulls real examples from the dataset to check
round-trip fidelity, so it needs network access the first time it isn't cached.

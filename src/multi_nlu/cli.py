"""Command line interface."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from multi_nlu import metrics, report
from multi_nlu.data import load_examples, sample_shots
from multi_nlu.formats import DEFAULT, format_names, get_format
from multi_nlu.predict import Prediction
from multi_nlu.report import Detail
from multi_nlu.schema import DatasetSpec, builtin_schemas, derive_schema, load_schema
from multi_nlu.tracking import Tracker

app = typer.Typer(help="Multi-intent NLU with generative language models.", no_args_is_help=True)

SchemaOpt = Annotated[
    str, typer.Option("--schema", "-s", help="Built-in schema name or YAML path.")
]
FormatOpt = Annotated[
    str, typer.Option("--format", "-f", help=f"Annotation format: {', '.join(format_names())}.")
]
ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Hugging Face model id.")]
SplitOpt = Annotated[str, typer.Option("--split", help="Dataset split to run on.")]
LimitOpt = Annotated[Optional[int], typer.Option("--limit", "-n", help="Random subsample size.")]
DetailOpt = Annotated[
    Detail,
    typer.Option(
        "--detail",
        "-d",
        help="Print each example: errors only, or all of them.",
        case_sensitive=False,
    ),
]


@app.command()
def predict(
    model: ModelOpt = "Qwen/Qwen3-1.7B",
    schema: SchemaOpt = "mixsnips",
    format: FormatOpt = DEFAULT,
    split: SplitOpt = "test",
    limit: LimitOpt = None,
    shots: Annotated[int, typer.Option(help="Few-shot demonstrations from the train split.")] = 0,
    adapter: Annotated[Optional[Path], typer.Option(help="LoRA adapter directory.")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 8,
    device: Annotated[
        Optional[str], typer.Option(help="cpu, mps or cuda; auto by default.")
    ] = None,
    out: Annotated[
        Optional[Path], typer.Option("--out", "-o", help="Write predictions as JSONL.")
    ] = None,
    detail: DetailOpt = Detail.NONE,
    seed: int = 0,
) -> None:
    """Annotate a split with a base or fine-tuned model, then score the result."""
    from multi_nlu.models import load_model
    from multi_nlu.predict import predict as run

    task, fmt = load_schema(schema), get_format(format)
    examples = load_examples(task, split, limit, seed)
    demos = sample_shots(load_examples(task, "train"), shots, seed) if shots else []

    model_, tokenizer, _ = load_model(
        model, adapter=str(adapter) if adapter else None, device=device
    )
    generations = run([e.text for e in examples], model_, tokenizer, task, fmt, demos, batch_size)
    preds = [
        Prediction(p.text, p.output, gold=fmt.render(e.annotation), format=fmt.name)
        for p, e in zip(generations, examples)
    ]

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(vars(p)) for p in preds) + "\n")
        typer.echo(f"wrote {len(preds)} predictions to {out}")

    typer.echo(_report(preds, detail))


@app.command()
def score(
    predictions: Annotated[Path, typer.Argument(help="JSONL written by `predict`.")],
    detail: DetailOpt = Detail.NONE,
) -> None:
    """Re-score a saved predictions file."""
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    typer.echo(_report([Prediction(**row) for row in rows], detail))


@app.command()
def train(
    model: ModelOpt = "Qwen/Qwen3-1.7B",
    schema: SchemaOpt = "mixsnips",
    format: FormatOpt = DEFAULT,
    out: Annotated[Path, typer.Option("--out", "-o", help="Adapter output directory.")] = Path(
        "runs/latest"
    ),
    limit: LimitOpt = None,
    eval_limit: Annotated[
        int, typer.Option("--eval-limit", help="Validation examples to track loss on.")
    ] = 200,
    epochs: float = 1.0,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 4,
    grad_accum: int = 4,
    lr: float = 1e-4,
    lora_r: int = 16,
    max_len: int = 1024,
    device: Annotated[Optional[str], typer.Option()] = None,
    tracker: Annotated[
        Tracker,
        typer.Option(help="Experiment tracker to report metrics to.", case_sensitive=False),
    ] = Tracker.NONE,
    run_name: Annotated[
        Optional[str], typer.Option(help="Run name for the tracker; defaults to --out's name.")
    ] = None,
    seed: int = 0,
) -> None:
    """Fine-tune a LoRA adapter on the train split."""
    from multi_nlu.train import TrainConfig
    from multi_nlu.train import train as run

    task = load_schema(schema)
    config = TrainConfig(
        model_id=model,
        out_dir=out,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        lora_r=lora_r,
        max_len=max_len,
        device=device,
        tracker=tracker,
        run_name=run_name,
        seed=seed,
    )
    run(
        task,
        get_format(format),
        load_examples(task, "train", limit, seed),
        load_examples(task, "validation", eval_limit, seed),
        config,
    )
    typer.echo(f"saved adapter to {out}")


@app.command(name="derive-schema")
def derive(
    dataset: Annotated[str, typer.Argument(help="Hugging Face dataset path.")],
    name: Annotated[
        str, typer.Option(help="Schema name; defaults to the dataset's last path part.")
    ] = "",
    split: SplitOpt = "train",
    text_column: str = "text",
    intents_column: str = "intents",
    min_freq: Annotated[
        float, typer.Option(help="Drop slots rarer than this within an intent.")
    ] = 0.01,
    out: Annotated[Optional[Path], typer.Option("--out", "-o")] = None,
) -> None:
    """Write a schema YAML by reading the annotations of a dataset."""
    spec = DatasetSpec(path=dataset, text_column=text_column, intents_column=intents_column)
    examples = load_examples(spec, split)
    result = derive_schema(
        (e.annotation for e in examples), name or dataset.rsplit("/", 1)[-1], spec, min_freq
    )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.to_yaml())
        typer.echo(f"wrote {out}")
    else:
        typer.echo(result.to_yaml())


@app.command()
def schemas() -> None:
    """List the bundled schemas."""
    for name in builtin_schemas():
        task = load_schema(name)
        typer.echo(f"{name}: {len(task.intents)} intents, {len(task.slots)} slots")


@app.command()
def formats() -> None:
    """List the annotation formats a model can be asked to produce."""
    for name in format_names():
        typer.echo(f"{name}: {get_format(name).description}")


@app.command()
def show(
    schema: SchemaOpt = "mixsnips",
    format: FormatOpt = DEFAULT,
    split: SplitOpt = "validation",
    n: Annotated[int, typer.Option("-n", help="Examples to print.")] = 3,
    prompt: Annotated[bool, typer.Option(help="Print the system prompt too.")] = False,
) -> None:
    """Print a few annotated examples, to sanity check a schema, format and data."""
    task, fmt = load_schema(schema), get_format(format)
    if prompt:
        typer.echo(fmt.system_prompt(task) + "\n")

    for example in load_examples(task, split, n):
        annotation = example.annotation
        typer.echo(example.text)
        typer.echo(fmt.render(annotation))
        for intent in annotation.intents:
            slots = ", ".join(f"{s.label}={s.text(annotation.text)!r}" for s in intent.slots)
            typer.echo(f"  {intent.label}: {slots}")
        typer.echo("")


def _report(preds: list[Prediction], detail: Detail = Detail.NONE) -> str:
    """The scores, optionally preceded by a diff of each example."""
    graded = [p for p in preds if p.gold is not None]
    if not graded:
        return "no gold annotations to score against"

    golds = [p.gold_annotation for p in graded]
    predicted = [p.annotation for p in graded]
    scores = metrics.score(golds, predicted, [p.text for p in graded])
    if detail is Detail.NONE:
        return scores.summary()

    diffs = [report.diff(g, a, p.text, p.output) for g, a, p in zip(golds, predicted, graded)]
    shown = [d for d in diffs if detail is Detail.ALL or d.status is not report.Status.CORRECT]
    return report.render_all(shown) + "\n".join([report.summarize(diffs), scores.summary()])


def main() -> None:
    app()

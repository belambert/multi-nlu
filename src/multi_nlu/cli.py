"""Command line interface."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from multi_nlu import metrics
from multi_nlu.annotation import parse_xml
from multi_nlu.data import load_examples, sample_shots
from multi_nlu.predict import Prediction
from multi_nlu.schema import DatasetSpec, builtin_schemas, derive_schema, load_schema

app = typer.Typer(help="Multi-intent NLU with generative language models.", no_args_is_help=True)

SchemaOpt = Annotated[
    str, typer.Option("--schema", "-s", help="Built-in schema name or YAML path.")
]
ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Hugging Face model id.")]
SplitOpt = Annotated[str, typer.Option("--split", help="Dataset split to run on.")]
LimitOpt = Annotated[Optional[int], typer.Option("--limit", "-n", help="Random subsample size.")]


@app.command()
def predict(
    model: ModelOpt = "Qwen/Qwen3-1.7B",
    schema: SchemaOpt = "mixsnips",
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
    seed: int = 0,
) -> None:
    """Tag a split with a base or fine-tuned model, then score the result."""
    from multi_nlu.models import load_model
    from multi_nlu.predict import predict as run

    task = load_schema(schema)
    examples = load_examples(task, split, limit, seed)
    demos = sample_shots(load_examples(task, "train"), shots, seed) if shots else []

    model_, tokenizer, _ = load_model(
        model, adapter=str(adapter) if adapter else None, device=device
    )
    preds = [
        Prediction(p.text, p.xml, gold=e.xml)
        for p, e in zip(
            run([e.text for e in examples], model_, tokenizer, task, demos, batch_size), examples
        )
    ]

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(vars(p)) for p in preds) + "\n")
        typer.echo(f"wrote {len(preds)} predictions to {out}")

    typer.echo(_score(preds))


@app.command()
def score(predictions: Annotated[Path, typer.Argument(help="JSONL written by `predict`.")]) -> None:
    """Re-score a saved predictions file."""
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    typer.echo(_score([Prediction(**row) for row in rows]))


@app.command()
def train(
    model: ModelOpt = "Qwen/Qwen3-1.7B",
    schema: SchemaOpt = "mixsnips",
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
        seed=seed,
    )
    run(
        task,
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
    target_column: str = "xml",
    min_freq: Annotated[
        float, typer.Option(help="Drop slots rarer than this within an intent.")
    ] = 0.01,
    out: Annotated[Optional[Path], typer.Option("--out", "-o")] = None,
) -> None:
    """Write a schema YAML by reading the annotations of a dataset."""
    spec = DatasetSpec(path=dataset, text_column=text_column, target_column=target_column)
    examples = load_examples(spec, split)
    result = derive_schema(
        (e.xml for e in examples), name or dataset.rsplit("/", 1)[-1], spec, min_freq
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
def show(
    schema: SchemaOpt = "mixsnips",
    split: SplitOpt = "validation",
    n: Annotated[int, typer.Option("-n", help="Examples to print.")] = 3,
    prompt: Annotated[bool, typer.Option(help="Print the system prompt too.")] = False,
) -> None:
    """Print a few annotated examples, to sanity check a schema and its data."""
    from multi_nlu.prompts import system_prompt

    task = load_schema(schema)
    if prompt:
        typer.echo(system_prompt(task) + "\n")

    for example in load_examples(task, split, n):
        annotation = example.annotation
        typer.echo(example.text)
        typer.echo(example.xml)
        for intent in annotation.intents:
            slots = ", ".join(f"{s.label}={s.text(annotation.text)!r}" for s in intent.slots)
            typer.echo(f"  {intent.label}: {slots}")
        typer.echo("")


def _score(preds: list[Prediction]) -> str:
    graded = [(p, p.gold) for p in preds if p.gold]
    if not graded:
        return "no gold annotations to score against"
    return metrics.score(
        [parse_xml(gold, p.text) for p, gold in graded],
        [p.annotation for p, _ in graded],
        [p.text for p, _ in graded],
    ).summary()


def main() -> None:
    app()

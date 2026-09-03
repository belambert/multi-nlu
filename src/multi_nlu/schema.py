"""Task schemas: the intents and slots of a dataset, loaded from YAML."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable

import yaml

BUILTIN_DIR = resources.files("multi_nlu") / "schemas"


@dataclass(frozen=True)
class DatasetSpec:
    """Where the annotated data lives and which columns hold it."""

    path: str
    text_column: str = "text"
    target_column: str = "xml"
    name: str | None = None  # HF config name, when the dataset has several


@dataclass(frozen=True)
class Schema:
    """The label inventory for one task, plus the dataset it was derived from."""

    name: str
    intents: dict[str, tuple[str, ...]]
    dataset: DatasetSpec | None = None
    description: str = ""
    source: Path | None = field(default=None, compare=False)

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(sorted({s for slots in self.intents.values() for s in slots}))

    def to_yaml(self) -> str:
        doc: dict = {"name": self.name}
        if self.description:
            doc["description"] = self.description
        if self.dataset:
            doc["dataset"] = {k: v for k, v in vars(self.dataset).items() if v is not None}
        doc["intents"] = {i: list(s) for i, s in self.intents.items()}
        return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


def load_schema(name_or_path: str | Path) -> Schema:
    """Load a bundled schema by name (e.g. "mixsnips") or a YAML file by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = Path(str(BUILTIN_DIR / f"{name_or_path}.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"no schema at {path}; built-ins: {builtin_schemas()}")

    doc = yaml.safe_load(path.read_text())
    dataset = doc.get("dataset")

    return Schema(
        name=doc.get("name", path.stem),
        intents={i: tuple(s or ()) for i, s in doc["intents"].items()},
        dataset=DatasetSpec(**dataset) if dataset else None,
        description=doc.get("description", ""),
        source=path,
    )


def builtin_schemas() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in BUILTIN_DIR.iterdir())


def derive_schema(
    xmls: Iterable[str],
    name: str,
    dataset: DatasetSpec | None = None,
    min_freq: float = 0.01,
) -> Schema:
    """Build a schema from gold annotations, dropping slots rarer than min_freq."""
    from multi_nlu.annotation import parse_xml

    totals: Counter[str] = Counter()
    seen: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for xml in xmls:
        for intent in parse_xml(xml).intents:
            totals[intent.label] += 1
            seen[intent.label].update(slot.label for slot in intent.slots)

    intents = {
        intent: tuple(sorted(s for s, n in seen[intent].items() if n / totals[intent] >= min_freq))
        for intent in sorted(totals)
    }
    return Schema(name=name, intents=intents, dataset=dataset)

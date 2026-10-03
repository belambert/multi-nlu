"""Per-example diffs: what one prediction got right, missed and invented.

The aggregate scores say how well a format did; these say where it went wrong,
which is what you read when deciding what to change.
"""

import shutil
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Hashable, Iterable, Sequence

import typer

from multi_nlu.annotation import Annotation, is_faithful
from multi_nlu.metrics import structure, value


class Status(StrEnum):
    CORRECT = "correct"  # every intent and slot matches
    PARTIAL = "partial"  # some of them do
    WRONG = "wrong"  # none do, but the output parsed
    MALFORMED = "malformed"  # the output did not parse


class Detail(StrEnum):
    NONE = "none"
    ERRORS = "errors"
    ALL = "all"


MARKS = {"ok": "✓", "missing": "-", "spurious": "+"}
LABEL = 9  # width of the status badge and the row labels, which share a column
HANG = " " * (LABEL + 3)
BADGES = {
    Status.CORRECT: "✓",
    Status.PARTIAL: "~",
    Status.WRONG: "✗",
    Status.MALFORMED: "!",
}
COLORS = {
    Status.CORRECT: typer.colors.GREEN,
    Status.PARTIAL: typer.colors.YELLOW,
    Status.WRONG: typer.colors.RED,
    Status.MALFORMED: typer.colors.MAGENTA,
}


@dataclass(frozen=True)
class Diff:
    """One prediction lined up against its gold annotation."""

    utterance: str
    status: Status
    intents: list[tuple[str, str]]  # (mark, display) in gold order, then spurious
    slots: list[tuple[str, str]]
    output: str = ""
    copied: str = ""  # the utterance as the prediction reproduced it, tags stripped
    faithful: bool = True

    def render(self) -> str:
        lines = [f"{_badge(self.status)} {self.utterance}"]
        if self.status is Status.MALFORMED:
            lines.append(_row("got", self.output))  # nothing parsed, so this is all there is
        elif not self.faithful:
            lines.extend(drift(self.utterance, self.copied))
        for name, items in (("intents", self.intents), ("slots", self.slots)):
            if items:
                lines.append(_row(name, *_cells(items)))
        return "\n".join(lines)


def _badge(status: Status) -> str:
    """The status mark and word, coloured; padded before styling to keep its width."""
    colour = COLORS[status]
    mark = typer.style(BADGES[status], fg=colour, bold=True)
    return f"{mark} {typer.style(f'{status:<{LABEL}}', fg=colour, bold=True)}"


def drift(utterance: str, copied: str) -> list[str]:
    """Two lines showing where a prediction stopped copying the utterance.

    The text the two share is dimmed so the words that differ carry the line:
    green for what the utterance says, red for what the prediction put there
    instead. Offsets measured through the red text no longer line up with gold.
    """
    want, got = utterance.split(), copied.split()
    left, right = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, want, got).get_opcodes():
        shared = tag == "equal"
        if i2 > i1:
            left.append(_segment(want[i1:i2], shared, typer.colors.GREEN))
        if j2 > j1:
            right.append(_segment(got[j1:j2], shared, typer.colors.RED))
    return [_row("want", " ".join(left)), _row("copied", " ".join(right))]


def _segment(words: Sequence[str], shared: bool, colour: str) -> str:
    text = " ".join(words)
    return typer.style(text, dim=True) if shared else typer.style(text, fg=colour)


def _row(label: str, *cells: str) -> str:
    """A labelled row; extra cells hang underneath, lined up with the first."""
    return f"  {label:<{LABEL}} " + f"\n{HANG}".join(cells)


def _cells(items: Sequence[tuple[str, str]]) -> list[str]:
    """Matched items collapse to one count, then an error each, so they line up."""
    matched = sum(mark == "ok" for mark, _ in items)
    counted = [f"{MARKS['ok']}{matched}"] if matched else []
    return counted + [f"{MARKS[m]}{d}" for m, d in items if m != "ok"]


def diff(gold: Annotation, pred: Annotation, utterance: str, output: str = "") -> Diff:
    """Compare one prediction to gold, marking every intent and slot."""
    intents = _mark(_intent_items(gold), _intent_items(pred))
    slots = _mark(_slot_items(gold), _slot_items(pred))

    return Diff(
        utterance=utterance,
        status=_status(gold, pred, intents + slots),
        intents=intents,
        slots=slots,
        output=output,
        copied=pred.text,
        faithful=is_faithful(pred, utterance),
    )


def render_all(diffs: Iterable[Diff]) -> str:
    """The diffs as one block, each ruled off from the next."""
    rule = "─" * min(shutil.get_terminal_size((80, 24)).columns, 88)
    return "".join(f"{d.render()}\n{rule}\n" for d in diffs)


def summarize(diffs: Sequence[Diff]) -> str:
    """A one-line count of each status."""
    counts = Counter(d.status for d in diffs)
    return "  ".join(f"{status} {counts[status]}" for status in Status if counts[status])


def _status(gold: Annotation, pred: Annotation, marked: list[tuple[str, str]]) -> Status:
    if pred.malformed:
        return Status.MALFORMED
    if structure(gold) == structure(pred):
        return Status.CORRECT
    return Status.PARTIAL if any(m == "ok" for m, _ in marked) else Status.WRONG


def _intent_items(a: Annotation) -> list[tuple[Hashable, str]]:
    return [(i.label, i.label) for i in a.intents]


def _slot_items(a: Annotation) -> list[tuple[Hashable, str]]:
    """Slots keyed as they are scored — scoped to their intent — and shown with
    their original casing."""
    return [
        ((i.label, s.label, value(s.text(a.text))), f"{i.label}.{s.label}={s.text(a.text)!r}")
        for i in a.intents
        for s in i.slots
    ]


def _mark(
    gold: Iterable[tuple[Hashable, str]], pred: Iterable[tuple[Hashable, str]]
) -> list[tuple[str, str]]:
    """Mark gold items ok or missing, then list what the prediction invented."""
    gold, pred = list(gold), list(pred)
    matched = Counter(k for k, _ in gold) & Counter(k for k, _ in pred)

    out, pool = [], Counter(matched)
    for key, display in gold:
        ok = bool(pool[key])
        pool[key] -= ok
        out.append(("ok" if ok else "missing", display))

    pool = Counter(matched)
    for key, display in pred:
        if pool[key]:
            pool[key] -= 1
        else:
            out.append(("spurious", display))
    return out

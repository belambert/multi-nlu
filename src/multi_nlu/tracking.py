"""Experiment tracker selection, kept free of heavy imports for fast CLI startup."""

from enum import StrEnum


class Tracker(StrEnum):
    """Experiment tracking backend to report training metrics to."""

    NONE = "none"
    WANDB = "wandb"
    TRACKIO = "trackio"


def check_tracker_available(tracker: Tracker) -> None:
    """Raise with an actionable message if tracker's package isn't installed."""
    if tracker is Tracker.NONE:
        return

    # import here, not at module level, so picking no tracker stays fast and dependency-free
    from transformers.integrations import is_trackio_available, is_wandb_available

    package, is_available = {
        Tracker.WANDB: ("wandb", is_wandb_available),
        Tracker.TRACKIO: ("trackio", is_trackio_available),
    }[tracker]
    if not is_available():
        raise RuntimeError(
            f"{package} is not installed; run `uv add {package}` "
            f"(or `uv sync --extra tracking`) to use --tracker {tracker.value}"
        )

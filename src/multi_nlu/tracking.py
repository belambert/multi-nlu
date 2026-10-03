"""Experiment tracker selection, kept free of heavy imports for fast CLI startup."""

import netrc
import os
from enum import StrEnum


class Tracker(StrEnum):
    """Experiment tracking backend to report training metrics to."""

    NONE = "none"
    WANDB = "wandb"
    TRACKIO = "trackio"


def _wandb_logged_in() -> bool:
    """Best-effort check for an already-configured wandb login (env var or .netrc)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").split("://", 1)[-1]
    try:
        return netrc.netrc(os.environ.get("NETRC")).authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def default_wandb_mode() -> None:
    """Fall back to offline wandb logging when the default tracker has no account configured.

    wandb is the default tracker, so a user who never ran `wandb login` shouldn't hit a login
    prompt (or a hard failure in a non-interactive run) just for running `train` out of the box.
    Does nothing if the user already set WANDB_MODE or has wandb credentials.
    """
    if "WANDB_MODE" in os.environ or _wandb_logged_in():
        return
    print("wandb: no account configured, logging offline (set WANDB_API_KEY to sync to the cloud)")
    os.environ["WANDB_MODE"] = "offline"


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

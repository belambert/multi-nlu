import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from multi_nlu.cli import app
from multi_nlu.tracking import Tracker, check_tracker_available, default_wandb_mode
from multi_nlu.train import TrainConfig

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip rich's ANSI styling, which can split a word across escape codes."""
    return _ANSI_RE.sub("", output)


def test_train_config_defaults_to_wandb():
    config = TrainConfig()

    assert config.tracker is Tracker.WANDB
    assert config.run_name is None


def test_train_config_accepts_tracker_and_run_name():
    config = TrainConfig(tracker=Tracker.WANDB, run_name="my-run")

    assert config.tracker is Tracker.WANDB
    assert config.run_name == "my-run"


def test_check_tracker_available_noop_for_none():
    check_tracker_available(
        Tracker.NONE
    )  # must not raise, must not import transformers.integrations


@patch("transformers.integrations.is_wandb_available", return_value=False)
def test_check_tracker_available_raises_when_wandb_missing(_mock):
    with pytest.raises(RuntimeError, match="uv add wandb"):
        check_tracker_available(Tracker.WANDB)


@patch("transformers.integrations.is_trackio_available", return_value=False)
def test_check_tracker_available_raises_when_trackio_missing(_mock):
    with pytest.raises(RuntimeError, match="uv add trackio"):
        check_tracker_available(Tracker.TRACKIO)


@patch("transformers.integrations.is_wandb_available", return_value=True)
def test_check_tracker_available_passes_when_installed(_mock):
    check_tracker_available(Tracker.WANDB)  # must not raise


def test_default_wandb_mode_falls_back_to_offline_without_credentials(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", "/nonexistent-netrc-file")

    default_wandb_mode()

    assert os.environ["WANDB_MODE"] == "offline"


def test_default_wandb_mode_leaves_api_key_login_alone(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    default_wandb_mode()

    assert "WANDB_MODE" not in os.environ


def test_default_wandb_mode_leaves_explicit_mode_alone(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    default_wandb_mode()

    assert os.environ["WANDB_MODE"] == "online"


def test_cli_rejects_invalid_tracker():
    result = runner.invoke(app, ["train", "--tracker", "not-a-tracker"])

    assert result.exit_code != 0
    assert "not-a-tracker" in _plain(result.output)


@pytest.mark.parametrize("tracker", ["none", "wandb", "trackio"])
def test_cli_accepts_valid_trackers(tracker):
    # fails past validation (no model/data fetch here), so just check it's not a usage error.
    # GitHub Actions makes typer force rich's terminal styling (it checks the GITHUB_ACTIONS env
    # var), which can split "--tracker" across ANSI escape codes mid-word; strip them before
    # asserting so this doesn't depend on the CI environment's rendering quirks
    result = runner.invoke(app, ["train", "--tracker", tracker, "--help"])

    assert result.exit_code == 0
    assert "--tracker" in _plain(result.output)


def test_cli_train_passes_tracker_and_run_name_through(tmp_path: Path):
    captured = {}

    def fake_run(schema, fmt, train_examples, eval_examples, config):
        captured["config"] = config
        return config.out_dir

    with (
        patch("multi_nlu.train.train", fake_run),
        patch("multi_nlu.cli.load_schema", return_value="schema"),
        patch("multi_nlu.cli.load_examples", return_value=[]),
        patch("multi_nlu.cli.get_format", return_value="format"),
    ):
        result = runner.invoke(
            app,
            [
                "train",
                "--tracker",
                "trackio",
                "--run-name",
                "custom-name",
                "--out",
                str(tmp_path / "adapter"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].tracker is Tracker.TRACKIO
    assert captured["config"].run_name == "custom-name"

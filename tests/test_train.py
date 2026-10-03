from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from multi_nlu.cli import app
from multi_nlu.tracking import Tracker, check_tracker_available
from multi_nlu.train import TrainConfig

runner = CliRunner()


def test_train_config_defaults_to_no_tracking():
    config = TrainConfig()

    assert config.tracker is Tracker.NONE
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


def test_cli_rejects_invalid_tracker():
    result = runner.invoke(app, ["train", "--tracker", "not-a-tracker"])

    assert result.exit_code != 0
    assert "not-a-tracker" in result.output


@pytest.mark.parametrize("tracker", ["none", "wandb", "trackio"])
def test_cli_accepts_valid_trackers(tracker):
    # fails past validation (no model/data fetch here), so just check it's not a usage error
    result = runner.invoke(app, ["train", "--tracker", tracker, "--help"])

    assert result.exit_code == 0
    assert "--tracker" in result.output


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

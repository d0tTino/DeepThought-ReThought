import importlib
from unittest import mock

import pytest

pytest.importorskip("torch")
train = importlib.import_module("deepthought.train")


def test_main_delegates_to_run_training(monkeypatch):
    dummy_run = mock.Mock(return_value=42)
    monkeypatch.setattr(train, "run_training", dummy_run)
    result = train.main(["--model-path", "mp", "--dataset-path", "ds"])
    dummy_run.assert_called_once()
    assert result == 42

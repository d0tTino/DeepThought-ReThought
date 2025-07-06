import importlib
import types
from unittest import mock

train_script = importlib.import_module('deepthought.train_script')


def test_run_delegates_to_train_module(monkeypatch):
    dummy_args = mock.Mock()
    dummy_run = mock.Mock(return_value=42)
    monkeypatch.setattr(train_script, 'train_utils', types.SimpleNamespace(run=dummy_run))
    result = train_script.run(dummy_args)
    dummy_run.assert_called_once_with(dummy_args)
    assert result == 42

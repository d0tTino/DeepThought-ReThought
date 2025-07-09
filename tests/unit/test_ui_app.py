import importlib
import os
import sys
import pytest


def test_ui_app_load(monkeypatch):
    """Ensure the Streamlit UI script imports without errors."""
    monkeypatch.setenv("MEMORY_API_URL", "http://test")
    sys.modules.pop("ui.app", None)
    pytest.importorskip("streamlit")
    importlib.import_module("ui.app")

import importlib
import os
import sys
from pathlib import Path

import pytest


def test_ui_app_load(monkeypatch):
    """Ensure the Streamlit UI script imports without errors."""
    monkeypatch.setenv("MEMORY_API_URL", "http://test")
    sys.modules.pop("ui.app", None)
    pytest.importorskip("streamlit")
    importlib.import_module("ui.app")


def test_ui_app_search(monkeypatch):
    """Interact with the Streamlit app and verify requests are made."""
    testing = pytest.importorskip("streamlit.testing.v1")
    import requests

    responses = []

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, json):
        responses.append((url, json))
        return DummyResp({"answer": "42"})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("MEMORY_API_URL", "http://api")

    app_path = Path(__file__).resolve().parents[2] / "ui" / "app.py"
    at = testing.AppTest.from_file(app_path)
    at.run()
    at.text_input[0].input("hello")
    at.button[0].click()
    at.run()

    assert responses == [("http://api/memory/query", {"query": "hello"})]
    assert at.subheader[0].value == "Retrieved Facts"

import json
import os
from pathlib import Path

import pytest

from deepthought.config import load_settings, Settings


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DT_NATS_URL", "nats://example:1234")
    monkeypatch.setenv("DT_DB__HOST", "db.example")
    monkeypatch.setenv("DT_MODEL_PATH", "a/model")
    monkeypatch.setenv("DT_MEMORY_FILE", "mem.txt")

    settings = load_settings()
    assert settings.nats_url == "nats://example:1234"
    assert settings.db.host == "db.example"
    assert settings.model_path == "a/model"
    assert settings.memory_file == "mem.txt"


def test_file_load(tmp_path):
    data = {
        "nats_url": "nats://file:4222",
        "db": {"user": "fileuser", "password": "p", "host": "dbfile", "port": 123},
        "model_path": "file/model",
        "memory_file": "filemem.json",
    }
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(data))

    settings = load_settings(str(cfg))
    assert settings.nats_url == "nats://file:4222"
    assert settings.db.port == 123
    assert settings.model_path == "file/model"
    assert settings.memory_file == "filemem.json"


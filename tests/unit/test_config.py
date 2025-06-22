import json

import yaml
import pytest

from deepthought.config import get_settings, load_settings


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


def test_yaml_file_load(tmp_path):
    data = {
        "nats_url": "nats://yaml:4222",
        "db": {"user": "yamluser", "password": "p", "host": "dbyaml", "port": 456},
        "model_path": "yaml/model",
        "memory_file": "yaml.json",
    }
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(data))

    settings = load_settings(str(cfg))
    assert settings.nats_url == "nats://yaml:4222"
    assert settings.db.port == 456
    assert settings.model_path == "yaml/model"
    assert settings.memory_file == "yaml.json"


def test_get_settings_reload(monkeypatch, tmp_path):
    cfg1 = tmp_path / "cfg1.json"
    cfg1.write_text(json.dumps({"nats_url": "nats://first"}))
    cfg2 = tmp_path / "cfg2.json"
    cfg2.write_text(json.dumps({"nats_url": "nats://second"}))

    monkeypatch.setenv("DT_CONFIG_FILE", str(cfg1))
    first = get_settings(str(cfg1))
    assert first.nats_url == "nats://first"

    monkeypatch.setenv("DT_CONFIG_FILE", str(cfg2))
    second = get_settings()
    assert second.nats_url == "nats://second"


def test_load_settings_empty_file(tmp_path):
    cfg = tmp_path / "empty.json"
    cfg.write_text("")
    with pytest.raises(ValueError):
        load_settings(str(cfg))


def test_load_settings_invalid_structure(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ValueError):
        load_settings(str(cfg))

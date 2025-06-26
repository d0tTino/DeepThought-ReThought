import json

import pytest
import yaml

from deepthought.config import get_settings, load_settings


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("DT_NATS_URL", "nats://example:1234")
    monkeypatch.setenv("DT_DB__HOST", "db.example")
    monkeypatch.setenv("DT_MODEL_PATH", "a/model")
    monkeypatch.setenv("DT_MEMORY_FILE", "mem.txt")
    monkeypatch.setenv("DT_REWARD__NOVELTY_THRESHOLD", "0.9")
    monkeypatch.setenv("DT_REWARD__SOCIAL_AFFINITY_THRESHOLD", "5")
    monkeypatch.setenv("DT_REWARD__WINDOW_SIZE", "10")
    monkeypatch.setenv("DT_REWARD__NOVELTY_WEIGHT", "1.5")
    monkeypatch.setenv("DT_REWARD__SOCIAL_WEIGHT", "2.0")
    monkeypatch.setenv("DT_REWARD__BUFFER_SIZE", "15")

    settings = load_settings()
    assert settings.nats_url == "nats://example:1234"
    assert settings.db.host == "db.example"
    assert settings.model_path == "a/model"
    assert settings.memory_file == "mem.txt"
    assert settings.reward.novelty_threshold == 0.9
    assert settings.reward.social_affinity_threshold == 5
    assert settings.reward.window_size == 10
    assert settings.reward.novelty_weight == 1.5
    assert settings.reward.social_weight == 2.0
    assert settings.reward.buffer_size == 15


def test_file_load(tmp_path):
    data = {
        "nats_url": "nats://file:4222",
        "db": {"user": "fileuser", "password": "p", "host": "dbfile", "port": 123},
        "model_path": "file/model",
        "memory_file": "filemem.json",
        "reward": {
            "novelty_threshold": 0.8,
            "social_affinity_threshold": 2,
            "window_size": 5,
            "novelty_weight": 1.1,
            "social_weight": 1.2,
            "buffer_size": 30,
        },
    }
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(data))

    settings = load_settings(str(cfg))
    assert settings.nats_url == "nats://file:4222"
    assert settings.db.port == 123
    assert settings.model_path == "file/model"
    assert settings.memory_file == "filemem.json"
    assert settings.reward.novelty_threshold == 0.8
    assert settings.reward.social_affinity_threshold == 2
    assert settings.reward.window_size == 5
    assert settings.reward.novelty_weight == 1.1
    assert settings.reward.social_weight == 1.2
    assert settings.reward.buffer_size == 30


def test_yaml_file_load(tmp_path):
    data = {
        "nats_url": "nats://yaml:4222",
        "db": {"user": "yamluser", "password": "p", "host": "dbyaml", "port": 456},
        "model_path": "yaml/model",
        "memory_file": "yaml.json",
        "reward": {
            "novelty_threshold": 0.7,
            "social_affinity_threshold": 3,
            "window_size": 7,
            "novelty_weight": 0.9,
            "social_weight": 0.8,
            "buffer_size": 25,
        },
    }
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(data))

    settings = load_settings(str(cfg))
    assert settings.nats_url == "nats://yaml:4222"
    assert settings.db.port == 456
    assert settings.model_path == "yaml/model"
    assert settings.memory_file == "yaml.json"
    assert settings.reward.novelty_threshold == 0.7
    assert settings.reward.social_affinity_threshold == 3
    assert settings.reward.window_size == 7
    assert settings.reward.novelty_weight == 0.9
    assert settings.reward.social_weight == 0.8
    assert settings.reward.buffer_size == 25


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

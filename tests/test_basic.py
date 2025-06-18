import json

from deepthought.config import load_settings


def test_load_settings_via_env_file(monkeypatch, tmp_path):
    """Settings should load from a config file referenced by DT_CONFIG_FILE."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"nats_url": "nats://envfile:4222", "db": {"host": "envdb"}}))
    monkeypatch.setenv("DT_CONFIG_FILE", str(cfg))

    settings = load_settings()
    assert settings.nats_url == "nats://envfile:4222"
    assert settings.db.host == "envdb"

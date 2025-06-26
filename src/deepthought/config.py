"""Central configuration for DeepThought reThought."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

try:  # YAML support is optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml may not be installed
    yaml = None


class DatabaseSettings(BaseSettings):
    """Database connection information."""

    host: str = "localhost"
    port: int = 5432
    user: str = "user"
    password: str = "password"
    name: str = "deepthought"


class RewardThresholds(BaseSettings):
    """Thresholds for scoring social and novelty rewards."""

    novelty_threshold: float = 0.3
    social_affinity_threshold: int = 1
    window_size: int = 20
    novelty_weight: float = 1.0
    social_weight: float = 1.0
    buffer_size: int = 50


class Settings(BaseSettings):
    """Application wide settings."""

    nats_url: str = "nats://localhost:4222"
    nats_stream_name: str = "deepthought_events"
    db: DatabaseSettings = DatabaseSettings()
    model_path: str = "distilgpt2"
    memory_file: str = "memory.json"
    reward: RewardThresholds = RewardThresholds()

    model_config = SettingsConfigDict(env_prefix="DT_", env_nested_delimiter="__")


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load settings from environment variables or a config file."""
    file_path = config_file or os.getenv("DT_CONFIG_FILE")
    if file_path:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError("Config file is empty")

        try:
            if path.suffix in {".yaml", ".yml"}:
                if not yaml:
                    raise RuntimeError("PyYAML required to load YAML config")
                data = yaml.safe_load(content)
            else:
                data = json.loads(content)
        except Exception as e:
            raise ValueError(f"Invalid config structure: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Config data must be a mapping")

        return Settings.model_validate(data)
    return Settings()


_settings_cache: Optional[Settings] = None
_settings_path: Optional[str] = None


def get_settings(config_file: Optional[str] = None) -> Settings:
    """Return cached :class:`Settings`, reloading if the config source changed."""
    global _settings_cache, _settings_path
    path = config_file or os.getenv("DT_CONFIG_FILE")
    if _settings_cache is None or path != _settings_path or config_file:
        _settings_cache = load_settings(config_file)
        _settings_path = path
    return _settings_cache

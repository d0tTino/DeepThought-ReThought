"""Central configuration for DeepThought reThought."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

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


class Settings(BaseSettings):
    """Application wide settings."""

    nats_url: str = "nats://localhost:4222"
    nats_stream_name: str = "deepthought_events"
    db: DatabaseSettings = DatabaseSettings()
    model_path: str = "distilgpt2"
    memory_file: str = "memory.json"

    class Config:
        env_prefix = "DT_"
        env_nested_delimiter = "__"


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load settings from environment variables or a config file."""
    file_path = config_file or os.getenv("DT_CONFIG_FILE")
    if file_path:
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix in {".yaml", ".yml"}:
                if not yaml:
                    raise RuntimeError("PyYAML required to load YAML config")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return Settings.parse_obj(data)
    return Settings()


# Module level singleton for convenience
settings = load_settings()

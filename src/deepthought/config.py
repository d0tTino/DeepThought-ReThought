"""Central configuration for DeepThought reThought."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field, ValidationError
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

    nats_url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    nats_stream_name: str = "deepthought_events"
    nats_tls_cert: str | None = os.getenv("NATS_TLS_CERT")
    nats_tls_key: str | None = os.getenv("NATS_TLS_KEY")
    nats_tls_ca: str | None = os.getenv("NATS_TLS_CA")
    nats_username: str | None = os.getenv("NATS_USERNAME")
    nats_password: str | None = os.getenv("NATS_PASSWORD")

    db: DatabaseSettings = DatabaseSettings()
    model_path: str = "distilgpt2"
    memory_file: str = "memory.json"
    search_db: str | None = None
    social_graph_db: str = os.getenv("SOCIAL_GRAPH_DB", "social_graph.db")

    vector_backend: str = Field("chroma", env="DT_VECTOR_BACKEND")
    vector_use_gpu: bool = Field(False, env="DT_VECTOR_USE_GPU")

    memory_capacity: int = 100
    memory_top_k: int = 3

    graph_backend: str = Field("memgraph", env="DT_GRAPH_BACKEND")
    mg_host: str = os.getenv("MG_HOST", "localhost")
    mg_port: int = int(os.getenv("MG_PORT", 7687))
    mg_user: str = os.getenv("MG_USER", "memgraph")
    mg_password: str = os.getenv("MG_PASSWORD", "memgraph")

    neo4j_host: str = Field(default_factory=lambda: os.getenv("NEO4J_HOST", "localhost"))
    neo4j_port: int = Field(default_factory=lambda: int(os.getenv("NEO4J_PORT", 7687)))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "neo4j"))

    reward: RewardThresholds = RewardThresholds()
    persona_descriptions: dict[str, str] = {
        "friendly": "You are a friendly assistant who responds warmly.",
        "playful": "You like to joke around in your answers.",
        "snarky": "You reply with terse, witty sarcasm.",
    }

    model_config = SettingsConfigDict(env_prefix="DT_", env_nested_delimiter="__")


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load settings from environment variables or a config file.

    Either the default ``DT_`` prefix or plain variable names are accepted
    for Memgraph options, e.g. ``DT_MG_HOST`` or ``MG_HOST``.
    """
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
        except RuntimeError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid config structure: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Config data must be a mapping")

        if hasattr(Settings, "model_validate"):
            return Settings.model_validate(data)

        # Fallback for environments where pydantic is stubbed during tests.
        inst = Settings()

        def _assign(obj: object, values: dict[str, object]) -> None:
            for key, val in values.items():
                if isinstance(val, dict):
                    sub = getattr(obj, key, None)
                    if sub is None:
                        setattr(obj, key, type("Sub", (), {})())
                        sub = getattr(obj, key)
                    _assign(sub, val)
                else:
                    setattr(obj, key, val)

        _assign(inst, data)
        return inst
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


class BotEnv(BaseSettings):
    """Environment variables required for running the Discord bot."""

    DISCORD_TOKEN: str
    MONITOR_CHANNEL: int
    NATS_URL: AnyUrl = "nats://localhost:4222"

    model_config = SettingsConfigDict(env_prefix="")


def load_bot_env() -> BotEnv:
    """Return bot environment settings or exit with a clear error."""

    try:
        return BotEnv()
    except ValidationError as exc:  # pragma: no cover - runtime validation
        missing = ", ".join(err["loc"][0] for err in exc.errors())
        raise SystemExit(f"Missing or invalid environment variables: {missing}") from exc

"""Central configuration for DeepThought reThought."""

from __future__ import annotations

import functools
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


def _apply_env_aliases() -> None:
    """Map legacy environment variables to the expected ``DT_*`` names."""

    mapping = {
        "MG_HOST": "DT_MG_HOST",
        "MG_PORT": "DT_MG_PORT",
        "MG_USER": "DT_MG_USER",
        "MG_PASSWORD": "DT_MG_PASSWORD",
        "NEO4J_HOST": "DT_NEO4J_HOST",
        "NEO4J_PORT": "DT_NEO4J_PORT",
        "NEO4J_USER": "DT_NEO4J_USER",
        "NEO4J_PASSWORD": "DT_NEO4J_PASSWORD",
        "SOCIAL_PERCEPTION_MODEL": "DT_SOCIAL_PERCEPTION_MODEL",
    }

    for old, new in mapping.items():
        if old in os.environ and new not in os.environ:
            os.environ[new] = os.environ[old]


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


DEFAULT_SOCIAL_MODEL = Path(__file__).resolve().parent / "perception" / "default_social_perception_model.json"


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
    social_perception_model: str = Field(str(DEFAULT_SOCIAL_MODEL), env="SOCIAL_PERCEPTION_MODEL")
    memory_file: str = "memory.json"
    search_db: str | None = None
    social_graph_db: str = os.getenv("SOCIAL_GRAPH_DB", "social_graph.db")

    # Weights & Biases integration
    wandb_enabled: bool = Field(False, env="DT_WANDB_ENABLED")
    wandb_project: str | None = Field(None, env="DT_WANDB_PROJECT")
    wandb_sweep_id: str | None = Field(None, env="DT_WANDB_SWEEP_ID")
    wandb_upload_artifacts: bool = Field(False, env="DT_WANDB_UPLOAD_ARTIFACTS")

    vector_backend: str = Field("chroma", env="DT_VECTOR_BACKEND")
    vector_use_gpu: bool = Field(False, env="DT_VECTOR_USE_GPU")

    memory_capacity: int = 100
    memory_top_k: int = 3

    graph_backend: str = Field("memgraph", env="DT_GRAPH_BACKEND")
    mg_host: str = Field("localhost", env="DT_MG_HOST")
    mg_port: int = Field(7687, env="DT_MG_PORT")
    mg_user: str = Field("memgraph", env="DT_MG_USER")
    mg_password: str = Field("memgraph", env="DT_MG_PASSWORD")

    neo4j_host: str = Field("localhost", env="DT_NEO4J_HOST")
    neo4j_port: int = Field(7687, env="DT_NEO4J_PORT")
    neo4j_user: str = Field("neo4j", env="DT_NEO4J_USER")
    neo4j_password: str = Field("neo4j", env="DT_NEO4J_PASSWORD")

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
    _apply_env_aliases()
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
        except Exception as e:  # pragma: no cover - unreachable in tests
            raise ValueError(f"Invalid config structure: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Config data must be a mapping")

        if hasattr(Settings, "model_validate"):
            return Settings.model_validate(data)

        # Fallback for environments where pydantic is stubbed during tests.
        inst = Settings()  # pragma: no cover

        def _assign(obj: object, values: dict[str, object]) -> None:  # pragma: no cover
            for key, val in values.items():
                if isinstance(val, dict):
                    sub = getattr(obj, key, None)
                    if sub is None:
                        setattr(obj, key, type("Sub", (), {})())
                        sub = getattr(obj, key)
                    _assign(sub, val)
                else:
                    setattr(obj, key, val)

        _assign(inst, data)  # pragma: no cover
        return inst  # pragma: no cover
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


def _first_non_empty(*names: str) -> str | None:
    """Return the first non-empty environment variable among ``names``."""

    for name in names:
        value = os.getenv(name)
        if value:
            value = value.strip()
            if value:
                return value
    return None


@functools.lru_cache(maxsize=1)
def get_git_commit() -> str | None:
    """Return the current git commit hash if available."""

    env_value = _first_non_empty("DT_GIT_COMMIT", "GIT_COMMIT", "SOURCE_COMMIT", "SOURCE_VERSION")
    if env_value:
        return env_value

    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(repo_root),
        )
    except Exception:  # pragma: no cover - git may be unavailable at runtime
        return None
    commit = result.stdout.strip()
    return commit or None


@functools.lru_cache(maxsize=1)
def get_package_version() -> str | None:
    """Return the installed ``deepthought`` package version if available."""

    env_value = _first_non_empty("DT_PACKAGE_VERSION", "PACKAGE_VERSION")
    if env_value:
        return env_value

    try:  # pragma: no cover - importlib metadata may not know about the package
        return metadata.version("deepthought")
    except metadata.PackageNotFoundError:
        pass
    except Exception:  # pragma: no cover - unexpected metadata failures
        return None

    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover - __version__ may be missing
        return None


@functools.lru_cache(maxsize=1)
def get_container_tag() -> str | None:
    """Return the container image tag if it was provided."""

    return _first_non_empty("DT_CONTAINER_TAG", "CONTAINER_TAG", "IMAGE_TAG")


class BotEnv(BaseSettings):
    """Environment variables required for running the Discord bot."""

    DISCORD_TOKEN: str
    MONITOR_CHANNEL: int
    PROJECTS_FORUM_CHANNEL: int | None = None
    PROJECTS_INDEX_CHANNEL: int | None = None
    PROJECTS_REQUIRE_EVENTS: bool = False
    NATS_URL: AnyUrl = "nats://localhost:4222"

    model_config = SettingsConfigDict(env_prefix="")


def load_bot_env() -> BotEnv:
    """Return bot environment settings or exit with a clear error."""

    try:
        return BotEnv()
    except ValidationError as exc:  # pragma: no cover - runtime validation
        missing = ", ".join(err["loc"][0] for err in exc.errors())
        raise SystemExit(f"Missing or invalid environment variables: {missing}") from exc

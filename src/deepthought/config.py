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

try:  # Pydantic v2 helper; optional for environments using stubs in tests
    from pydantic import AliasChoices, model_validator
except (ImportError, AttributeError):  # pragma: no cover - exercised in stubbed tests
    AliasChoices = None  # type: ignore[assignment]

    def model_validator(*args: object, **kwargs: object):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator


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


def _assign_settings(obj: object, values: dict[str, object]) -> None:
    """Recursively assign ``values`` onto ``obj`` for stubbed settings classes."""

    for key, val in values.items():
        if isinstance(obj, dict):
            if isinstance(val, dict):
                sub = obj.get(key)
                if not isinstance(sub, dict):
                    sub = {}
                    obj[key] = sub
                _assign_settings(sub, val)
            else:
                obj[key] = val
            continue
        if isinstance(val, dict):
            sub = getattr(obj, key, None)
            if sub is None:
                setattr(obj, key, type("Sub", (), {})())
                sub = getattr(obj, key)
            _assign_settings(sub, val)
        else:
            setattr(obj, key, val)


def _build_env_overrides() -> dict[str, object]:
    """Collect environment overrides when Pydantic settings are stubbed."""

    overrides: dict[str, object] = {}

    def _set(path: tuple[str, ...], value: object | None) -> None:
        if value in (None, ""):
            return
        target: dict[str, object] = overrides
        for key in path[:-1]:
            target = target.setdefault(key, {})  # type: ignore[assignment]
        target[path[-1]] = value

    def _coerce_int(value: str | None) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _coerce_float(value: str | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _coerce_bool(value: str | None) -> bool | None:
        if value in (None, ""):
            return None
        lowered = value.strip().lower()
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}
        if lowered in truthy:
            return True
        if lowered in falsy:
            return False
        return None

    _set(("nats_url",), os.getenv("DT_NATS_URL"))
    _set(("model_path",), os.getenv("DT_MODEL_PATH"))
    _set(("memory_file",), os.getenv("DT_MEMORY_FILE"))
    _set(("vector_backend",), os.getenv("DT_VECTOR_BACKEND"))
    vector_gpu = _coerce_bool(os.getenv("DT_VECTOR_USE_GPU"))
    if vector_gpu is not None:
        _set(("vector_use_gpu",), vector_gpu)
    _set(("graph_backend",), os.getenv("DT_GRAPH_BACKEND"))

    response_filter_enabled = _coerce_bool(os.getenv("DT_RESPONSE_FILTER_ENABLED"))
    if response_filter_enabled is not None:
        _set(("response_filter_enabled",), response_filter_enabled)
    _set(("response_filter_classifier",), os.getenv("DT_RESPONSE_FILTER_CLASSIFIER"))
    _set(("response_filter_fallback_message",), os.getenv("DT_RESPONSE_FILTER_FALLBACK_MESSAGE"))
    response_filter_denylist_raw = os.getenv("DT_RESPONSE_FILTER_DENYLIST")
    if response_filter_denylist_raw:
        try:
            parsed = json.loads(response_filter_denylist_raw)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in response_filter_denylist_raw.split(",")]
        if isinstance(parsed, list):
            _set(("response_filter_denylist",), parsed)

    _set(("db", "host"), os.getenv("DT_DB__HOST"))
    db_port = _coerce_int(os.getenv("DT_DB__PORT"))
    if db_port is not None:
        _set(("db", "port"), db_port)
    _set(("db", "user"), os.getenv("DT_DB__USER"))
    _set(("db", "password"), os.getenv("DT_DB__PASSWORD"))
    _set(("db", "name"), os.getenv("DT_DB__NAME"))

    reward_paths = {
        "DT_REWARD__NOVELTY_THRESHOLD": ("reward", "novelty_threshold"),
        "DT_REWARD__SOCIAL_AFFINITY_THRESHOLD": ("reward", "social_affinity_threshold"),
        "DT_REWARD__WINDOW_SIZE": ("reward", "window_size"),
        "DT_REWARD__NOVELTY_WEIGHT": ("reward", "novelty_weight"),
        "DT_REWARD__SOCIAL_WEIGHT": ("reward", "social_weight"),
        "DT_REWARD__BUFFER_SIZE": ("reward", "buffer_size"),
    }
    for env_name, path in reward_paths.items():
        raw = os.getenv(env_name)
        if path[-1] in {"novelty_threshold", "novelty_weight", "social_weight"}:
            coerced = _coerce_float(raw)
        else:
            coerced = _coerce_int(raw)
        if coerced is not None:
            _set(path, coerced)

    return overrides

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
    persona_sentiment_weight: float = Field(0.0, env="DT_PERSONA_SENTIMENT_WEIGHT")

    response_filter_enabled: bool = Field(True, env="DT_RESPONSE_FILTER_ENABLED")
    response_filter_denylist: list[str] = Field(default_factory=list, env="DT_RESPONSE_FILTER_DENYLIST")
    response_filter_classifier: str | None = Field(None, env="DT_RESPONSE_FILTER_CLASSIFIER")
    response_filter_fallback_message: str = Field(
        "I'm sorry, I can't assist with that.",
        env="DT_RESPONSE_FILTER_FALLBACK_MESSAGE",
    )

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
        _assign_settings(inst, data)  # pragma: no cover
        return inst  # pragma: no cover
    if hasattr(Settings, "model_validate"):
        return Settings()
    inst = Settings()  # pragma: no cover - executed with stubbed settings
    env_overrides = _build_env_overrides()
    if env_overrides:
        _assign_settings(inst, env_overrides)
    return inst


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


def _alias_choices(*names: str) -> object:
    """Return a ``validation_alias`` compatible value across environments."""

    if AliasChoices is not None:  # pragma: no branch - trivial branch
        return AliasChoices(*names)
    # Fallback for stubbed pydantic used in unit tests. Returning the first
    # name keeps type checking simple while ``load_bot_env`` injects aliases.
    return names[0]


class BotEnv(BaseSettings):
    """Environment variables required for running the Discord bot."""

    DISCORD_TOKEN: str
    MONITOR_CHANNEL: int
    PROJECT_FORUM_CHANNEL_ID: int | None = Field(
        default=None,
        validation_alias=_alias_choices("PROJECT_FORUM_CHANNEL_ID", "PROJECTS_FORUM_CHANNEL"),
    )
    PROJECT_INDEX_CHANNEL_ID: int | None = Field(
        default=None,
        validation_alias=_alias_choices("PROJECT_INDEX_CHANNEL_ID", "PROJECTS_INDEX_CHANNEL"),
    )
    PROJECT_REQUIRE_EVENTS: bool = Field(
        default=False,
        validation_alias=_alias_choices("PROJECT_REQUIRE_EVENTS", "PROJECTS_REQUIRE_EVENTS"),
    )
    PROJECT_HOLIDAY_LOCALE: str = Field(
        default="US",
        validation_alias=_alias_choices(
            "PROJECT_HOLIDAY_LOCALE", "PROJECTS_HOLIDAY_LOCALE"
        ),
    )
    NATS_URL: AnyUrl = "nats://localhost:4222"

    model_config = SettingsConfigDict(env_prefix="")

    @model_validator(mode="before")
    @classmethod
    def _promote_legacy_fields(cls, data: object) -> object:
        """Map legacy field names to their canonical counterparts."""

        if not isinstance(data, dict):
            return data

        updated = dict(data)

        def _copy_first(target: str, aliases: tuple[str, ...]) -> None:
            if updated.get(target) is not None:
                return
            for alias in aliases:
                value = updated.get(alias)
                if value not in (None, ""):
                    updated[target] = value
                    break

        _copy_first("PROJECT_FORUM_CHANNEL_ID", ("PROJECTS_FORUM_CHANNEL_ID", "PROJECTS_FORUM_CHANNEL"))
        _copy_first("PROJECT_INDEX_CHANNEL_ID", ("PROJECTS_INDEX_CHANNEL_ID", "PROJECTS_INDEX_CHANNEL"))
        _copy_first("PROJECT_REQUIRE_EVENTS", ("PROJECTS_REQUIRE_EVENTS",))
        _copy_first("PROJECT_HOLIDAY_LOCALE", ("PROJECTS_HOLIDAY_LOCALE",))

        return updated

    @property
    def PROJECTS_FORUM_CHANNEL(self) -> int | None:  # pragma: no cover - compatibility shim
        return self.PROJECT_FORUM_CHANNEL_ID

    @property
    def PROJECTS_INDEX_CHANNEL(self) -> int | None:  # pragma: no cover - compatibility shim
        return self.PROJECT_INDEX_CHANNEL_ID

    @property
    def PROJECTS_REQUIRE_EVENTS(self) -> bool:  # pragma: no cover - compatibility shim
        return self.PROJECT_REQUIRE_EVENTS



def load_bot_env() -> BotEnv:
    """Return bot environment settings or exit with a clear error."""

    for legacy, canonical in (
        ("PROJECTS_FORUM_CHANNEL", "PROJECT_FORUM_CHANNEL_ID"),
        ("PROJECTS_INDEX_CHANNEL", "PROJECT_INDEX_CHANNEL_ID"),
        ("PROJECTS_REQUIRE_EVENTS", "PROJECT_REQUIRE_EVENTS"),
    ):
        if legacy in os.environ and canonical not in os.environ:
            os.environ[canonical] = os.environ[legacy]

    try:
        env = BotEnv()
    except ValidationError as exc:  # pragma: no cover - runtime validation
        missing = ", ".join(err["loc"][0] for err in exc.errors())
        raise SystemExit(f"Missing or invalid environment variables: {missing}") from exc

    is_stub = BotEnv.__mro__ == (BotEnv, object)
    if is_stub:
        missing: list[str] = []
        invalid: list[str] = []

        token = os.getenv("DISCORD_TOKEN")
        if not token:
            missing.append("DISCORD_TOKEN")

        monitor_raw = os.getenv("MONITOR_CHANNEL")
        monitor: int | None = None
        if monitor_raw in (None, ""):
            missing.append("MONITOR_CHANNEL")
        else:
            try:
                monitor = int(monitor_raw)
            except ValueError:
                invalid.append("MONITOR_CHANNEL")

        def _optional_int(name: str) -> int | None:
            raw = os.getenv(name)
            if raw in (None, ""):
                return None
            try:
                return int(raw)
            except ValueError:
                invalid.append(name)
                return None

        def _coerce_bool(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            if raw in (None, ""):
                return default
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            invalid.append(name)
            return default

        forum_id = _optional_int("PROJECT_FORUM_CHANNEL_ID")
        index_id = _optional_int("PROJECT_INDEX_CHANNEL_ID")
        require_events = _coerce_bool("PROJECT_REQUIRE_EVENTS")
        locale = os.getenv("PROJECT_HOLIDAY_LOCALE", "US") or "US"
        nats_url = os.getenv("NATS_URL", "nats://localhost:4222")

        if missing or invalid:
            problems = sorted({*missing, *invalid})
            raise SystemExit(
                f"Missing or invalid environment variables: {', '.join(problems)}"
            )

        setattr(env, "DISCORD_TOKEN", token)
        setattr(env, "MONITOR_CHANNEL", monitor)
        setattr(env, "PROJECT_FORUM_CHANNEL_ID", forum_id)
        setattr(env, "PROJECT_INDEX_CHANNEL_ID", index_id)
        setattr(env, "PROJECT_REQUIRE_EVENTS", require_events)
        setattr(env, "PROJECT_HOLIDAY_LOCALE", locale.upper())
        setattr(env, "NATS_URL", nats_url)
        return env

    locale = getattr(env, "PROJECT_HOLIDAY_LOCALE", "US")
    normalized = str(locale).upper()
    if normalized != locale:
        setattr(env, "PROJECT_HOLIDAY_LOCALE", normalized)
    return env

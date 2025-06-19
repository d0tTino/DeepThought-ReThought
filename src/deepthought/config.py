"""Default configuration for the DeepThought reThought project."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os


@dataclass
class DeepThoughtConfig:
    """Configuration values for connecting to NATS and JetStream."""

    #: URL of the NATS server used for tests and local development
    nats_url: str = "nats://localhost:4222"

    #: Default JetStream stream name for DeepThought events
    stream_name: str = "deepthought_events"

    def as_dict(self) -> dict[str, str]:
        """Return the configuration as a dictionary."""
        return asdict(self)


def load_config_from_env() -> DeepThoughtConfig:
    """Load configuration values, falling back to defaults."""

    return DeepThoughtConfig(
        nats_url=os.getenv("NATS_URL", DeepThoughtConfig.nats_url),
        stream_name=os.getenv("STREAM_NAME", DeepThoughtConfig.stream_name),
    )


#: The configuration instance used by the library and tests
DEFAULT_CONFIG: DeepThoughtConfig = load_config_from_env()

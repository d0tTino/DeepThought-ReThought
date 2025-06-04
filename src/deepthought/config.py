# Basic configuration for DeepThought reThought
# Provides defaults for connecting to NATS and JetStream.

from dataclasses import dataclass, field
import os

@dataclass
class NATSSettings:
    """Configuration for NATS connectivity."""
    url: str = os.getenv("NATS_URL", "nats://localhost:4222")
    stream_name: str = os.getenv("NATS_STREAM_NAME", "deepthought_events")

@dataclass
class Settings:
    """Application wide settings."""
    nats: NATSSettings = field(default_factory=NATSSettings)

# Module level singleton for convenience
settings = Settings()

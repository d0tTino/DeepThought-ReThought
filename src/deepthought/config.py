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
class MemorySettings:
    """Configuration for BasicMemory module."""
    file_path: str = os.getenv("BASIC_MEMORY_FILE", "memory.json")

@dataclass
class Settings:
    """Application wide settings."""
    nats: NATSSettings = field(default_factory=NATSSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)

# Module level singleton for convenience
settings = Settings()

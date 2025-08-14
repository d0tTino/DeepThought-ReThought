"""Configuration for the perception service."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class PerceptionConfig(BaseSettings):
    """Settings controlling the perception service."""

    nats_url: str = Field("nats://localhost:4222", env="DT_NATS_URL")

    class Config:
        env_prefix = "DT_PERCEPTION_"

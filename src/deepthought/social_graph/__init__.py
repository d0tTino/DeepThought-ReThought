from __future__ import annotations

from . import constants
from . import db_manager as db_module
from .constants import *  # noqa: F401,F403
from .db_manager import *  # noqa: F401,F403
from .idle import _get_idle_generator, generate_idle_response
from .service import SocialGraphService
from .utils import analyze_sentiment

try:
    from examples import social_graph_bot as _bot_mod
except Exception:  # pragma: no cover - optional example missing deps
    _bot_mod = None

_nats_client = None
_js_context = None
_input_publisher = None


async def send_to_prism(data: dict) -> None:
    if _bot_mod and hasattr(_bot_mod, "send_to_prism"):
        await _bot_mod.send_to_prism(data)


async def publish_input_received(text: str) -> None:
    if _bot_mod and hasattr(_bot_mod, "publish_input_received"):
        await _bot_mod.publish_input_received(text)


def evaluate_triggers(message):
    if _bot_mod and hasattr(_bot_mod, "evaluate_triggers"):
        return _bot_mod.evaluate_triggers(message)
    return []


__all__ = [
    "SocialGraphService",
    "generate_idle_response",
    "_get_idle_generator",
    "send_to_prism",
    "publish_input_received",
    "evaluate_triggers",
    "analyze_sentiment",
    *constants.__all__,  # type: ignore[list-item]
    *db_module.__all__,  # type: ignore[list-item]
]

from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, Optional

from .trust_service import TrustService


class EngagementPolicy:
    """Determine reply strategy based on trust and activity signals.

    The policy examines a user's trust score, recent reply history and recent
    bot activity to choose between three response modes: ``"silent"`` (no
    reply), ``"minimal"`` (acknowledgement only) and ``"full"`` (complete
    response).

    Configuration options
    ---------------------
    full_threshold:
        Minimum trust score required for a full response.
    minimal_threshold:
        Minimum trust score required for any response. Scores below this value
        result in ``"silent"``.
    cooldown_seconds:
        Number of seconds that must elapse between replies to the same user.
    bot_cooldown:
        Seconds to wait after detecting a message from another bot before
        responding. Helps prevent bot-to-bot chatter.
    """

    def __init__(
        self,
        trust_service: TrustService | None = None,
        *,
        full_threshold: float = 0.0,
        minimal_threshold: float = -5.0,
        cooldown_seconds: float = 0.0,
        bot_cooldown: float = 0.0,
    ) -> None:
        self._trust_service = trust_service
        self.full_threshold = float(full_threshold)
        self.minimal_threshold = float(minimal_threshold)
        self.cooldown = max(float(cooldown_seconds), 0.0)
        self.bot_cooldown = max(float(bot_cooldown), 0.0)
        self._last_reply: Dict[str | int, datetime] = {}
        self._last_bot_message: Optional[datetime] = None

    async def response_mode(self, message) -> str:
        """Return the appropriate response mode for ``message``."""

        now = datetime.now(UTC)
        author = getattr(message, "author", None)
        user_id = getattr(author, "id", None)
        is_bot = bool(getattr(author, "bot", False))

        if is_bot:
            self._last_bot_message = now
            return "silent"

        if self._last_bot_message and (now - self._last_bot_message).total_seconds() < self.bot_cooldown:
            return "silent"

        if user_id is not None and self.cooldown > 0:
            last = self._last_reply.get(user_id)
            if last and (now - last).total_seconds() < self.cooldown:
                return "silent"

        trust = 0.0
        if self._trust_service and user_id is not None:
            try:
                trust = await self._trust_service.get_trust(user_id)
            except Exception:
                trust = 0.0

        if trust < self.minimal_threshold:
            mode = "silent"
        elif trust < self.full_threshold:
            mode = "minimal"
        else:
            mode = "full"

        if mode != "silent" and user_id is not None:
            self._last_reply[user_id] = now
        return mode

    async def should_reply(self, message) -> bool:
        """Return ``True`` if ``message`` warrants a reply."""

        return (await self.response_mode(message)) != "silent"


_default_policy: EngagementPolicy | None = None


async def should_reply(message) -> bool:
    """Convenience wrapper using a default :class:`EngagementPolicy`."""

    global _default_policy
    if _default_policy is None:
        _default_policy = EngagementPolicy()
    return await _default_policy.should_reply(message)

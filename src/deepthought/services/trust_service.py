from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Dict

from .db_manager import DBManager


class TrustService:
    """Manage per-user trust scores with optional exponential decay.

    Parameters
    ----------
    db_manager:
        Instance of :class:`DBManager` used for persistence. If not provided, a
        new :class:`DBManager` is created.
    decay:
        Exponential decay rate per second. A value of ``0`` disables decay.
    """

    def __init__(self, db_manager: DBManager | None = None, *, decay: float = 0.0) -> None:
        self._db = db_manager or DBManager()
        self.decay = max(decay, 0.0)
        self._last_update: Dict[str | int, datetime] = {}

    async def _apply_decay(self, user_id: str | int) -> float:
        """Return the current trust score after applying decay.

        The decayed value is written back to the database so that future calls
        operate on the updated value.
        """

        now = datetime.now(UTC)
        score = await self._db.get_trust(user_id)
        last = self._last_update.get(user_id, now)
        elapsed = (now - last).total_seconds()
        if self.decay > 0 and elapsed > 0 and score:
            factor = math.exp(-self.decay * elapsed)
            decayed = score * factor
            await self._db.adjust_trust(user_id, decayed - score)
            score = decayed
        self._last_update[user_id] = now
        return float(score)

    async def adjust_trust(self, user_id: str | int, delta: float) -> float:
        """Adjust ``user_id``'s trust score by ``delta`` and return the new score."""

        score = await self._apply_decay(user_id)
        await self._db.adjust_trust(user_id, float(delta))
        return score + float(delta)

    async def get_trust(self, user_id: str | int) -> float:
        """Retrieve ``user_id``'s trust score with decay applied."""

        return await self._apply_decay(user_id)

    async def is_trusted(self, user_id: str | int, threshold: float) -> bool:
        """Return ``True`` if ``user_id``'s trust meets or exceeds ``threshold``."""

        return (await self.get_trust(user_id)) >= float(threshold)

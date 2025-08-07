from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Dict, Tuple

from .db_manager import DBManager


class TrustService:
    """Manage per-user trust scores with optional exponential decay.

    Parameters
    ----------
    db_manager:
        Instance of :class:`DBManager` used for persistence. If not provided, a
        new :class:`DBManager` is created.
    decay:
        Exponential decay rate per second for overall trust. A value of ``0``
        disables decay.
    manipulative_penalty:
        Base trust deduction applied for a manipulative action. The actual
        penalty scales with the decayed offense count.
    manipulative_decay:
        Exponential decay rate per second applied to the manipulative offense
        counter. Higher values mean faster cooldown between offenses.
    """

    def __init__(
        self,
        db_manager: DBManager | None = None,
        *,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
        decay: float | None = None,
        manipulative_penalty: float = 0.1,
        manipulative_decay: float = 0.0,
    ) -> None:
        self._db = db_manager or DBManager()
        self.lower_limit = lower_limit if lower_limit is not None else -10.0
        self.upper_limit = upper_limit if upper_limit is not None else 10.0
        self.decay = max(decay or 0.0, 0.0)
        self.manipulative_penalty = float(manipulative_penalty)
        self.manipulative_decay = max(float(manipulative_decay), 0.0)
        self._init_params: Tuple[float | None, float | None, float | None] = (
            lower_limit,
            upper_limit,
            decay,
        )
        self._params_loaded = False
        self._last_update: Dict[str | int, datetime] = {}
        self._manipulative_state: Dict[str | int, Tuple[float, datetime]] = {}

    async def _load_params(self) -> None:
        if not self._params_loaded:
            lower, upper, decay = await self._db.get_trust_params()
            init_lower, init_upper, init_decay = self._init_params
            if any(v is not None for v in self._init_params):
                lower = init_lower if init_lower is not None else lower
                upper = init_upper if init_upper is not None else upper
                decay = init_decay if init_decay is not None else decay
                await self._db.set_trust_params(lower, upper, decay)
            self.lower_limit, self.upper_limit, self.decay = lower, upper, max(decay, 0.0)
            self._params_loaded = True

    async def _apply_decay(self, user_id: str | int) -> float:
        """Return the current trust score after applying decay.

        The decayed value is written back to the database so that future calls
        operate on the updated value.
        """

        await self._load_params()
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
        new_score = score + float(delta)
        if new_score < self.lower_limit:
            new_score = self.lower_limit
        elif new_score > self.upper_limit:
            new_score = self.upper_limit
        await self._db.adjust_trust(user_id, new_score - score)
        return new_score

    async def get_trust(self, user_id: str | int) -> float:
        """Retrieve ``user_id``'s trust score with decay applied."""

        return await self._apply_decay(user_id)

    async def is_trusted(self, user_id: str | int, threshold: float) -> bool:
        """Return ``True`` if ``user_id``'s trust meets or exceeds ``threshold``."""

        return (await self.get_trust(user_id)) >= float(threshold)

    async def penalize_manipulative(self, user_id: str | int) -> float:
        """Apply a manipulative penalty with exponential cooldown."""

        await self._db.increment_offense(user_id, "manipulative")
        now = datetime.now(UTC)
        severity, last = self._manipulative_state.get(user_id, (0.0, now))
        elapsed = (now - last).total_seconds()
        if self.manipulative_decay > 0 and elapsed >= 1.0 and severity:
            severity *= math.exp(-self.manipulative_decay * elapsed)
        severity += 1.0
        self._manipulative_state[user_id] = (severity, now)
        penalty = -self.manipulative_penalty * severity
        return await self.adjust_trust(user_id, penalty)

    async def penalize_banned(self, user_id: str | int) -> float:
        count = await self._db.increment_offense(user_id, "banned")
        return await self.adjust_trust(user_id, -1.0 * count)

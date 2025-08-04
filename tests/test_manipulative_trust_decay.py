import math
from datetime import timedelta

import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager
from deepthought.services.trust_service import TrustService


@pytest.mark.asyncio
async def test_manipulative_offense_cooldown(tmp_path):
    db = DBManager(str(tmp_path / "db.sqlite"))
    await db.init_db()
    service = TrustService(db_manager=db, manipulative_penalty=0.1, manipulative_decay=1.0)
    user = "u1"

    score1 = await service.penalize_manipulative(user)
    assert score1 == pytest.approx(-0.1, rel=1e-3)

    score2 = await service.penalize_manipulative(user)
    assert score2 == pytest.approx(-0.3, abs=1e-3)

    severity, last = service._manipulative_state[user]
    service._manipulative_state[user] = (severity, last - timedelta(seconds=5))

    score3 = await service.penalize_manipulative(user)
    expected_severity = severity * math.exp(-1.0 * 5) + 1
    expected_penalty = -0.1 * expected_severity
    assert score3 == pytest.approx(-0.3 + expected_penalty, abs=1e-3)

    await db.close()

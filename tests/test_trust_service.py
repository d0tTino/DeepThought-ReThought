import math
from datetime import timedelta

import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager
from deepthought.services.trust_service import TrustService


@pytest.mark.asyncio
async def test_decay_and_limits(tmp_path):
    db = DBManager(str(tmp_path / "db.sqlite"))
    await db.init_db()
    await db.set_trust_params(-2.0, 2.0, 0.1)
    service = TrustService(db_manager=db)

    assert await service.adjust_trust("u1", 5.0) == 2.0
    assert await service.adjust_trust("u1", -10.0) == -2.0

    service._last_update["u1"] -= timedelta(seconds=10)
    decayed = await service.get_trust("u1")
    expected = -2.0 * math.exp(-0.1 * 10)
    assert pytest.approx(decayed, rel=1e-3) == expected

    await db.close()


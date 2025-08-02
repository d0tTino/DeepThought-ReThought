import math
from datetime import timedelta

import pytest

pytest.importorskip("aiosqlite")

from deepthought.services.db_manager import DBManager
from deepthought.services.trust_service import TrustService


@pytest.mark.asyncio
async def test_trust_decay_and_threshold(tmp_path):
    db = DBManager(str(tmp_path / "db.sqlite"))
    await db.init_db()
    service = TrustService(db_manager=db, decay=0.1)

    assert pytest.approx(await service.adjust_trust("u1", 1.0)) == 1.0
    assert await service.is_trusted("u1", 0.5)

    service._last_update["u1"] -= timedelta(seconds=10)
    decayed = await service.get_trust("u1")
    expected = math.exp(-0.1 * 10)
    assert pytest.approx(decayed, rel=1e-3) == expected
    assert not await service.is_trusted("u1", 1.0)

    await db.close()

from __future__ import annotations

from dataclasses import dataclass

import deepthought.quest.writer as writer_module
from deepthought.quest.writer import QuestWriter


@dataclass
class Quest:
    id: int
    name: str
    description: str


def test_send_board_update_success(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):  # pragma: no cover - mocked
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

    monkeypatch.setattr(writer_module.requests, "post", fake_post)

    quest = Quest(1, "quest", "desc")
    writer = QuestWriter(board_channel="123", token="tok")

    writer.send_board_update(quest, event="start")

    assert "channels/123/messages" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bot tok"
    assert captured["json"]["content"].startswith("[start] quest")


def test_send_board_update_missing_token(monkeypatch):
    called = False

    def fake_post(*args, **kwargs):  # pragma: no cover - mocked
        nonlocal called
        called = True

    monkeypatch.setattr(writer_module.requests, "post", fake_post)

    quest = Quest(1, "quest", "desc")
    writer = QuestWriter(board_channel="123", token=None)

    writer.send_board_update(quest)

    assert not called


def test_send_board_update_network_failure(monkeypatch):
    def fake_post(*args, **kwargs):  # pragma: no cover - mocked
        raise RuntimeError("boom")

    warnings: list[str] = []

    def fake_warning(msg, *args, **kwargs):  # pragma: no cover - mocked
        warnings.append(msg)

    monkeypatch.setattr(writer_module.requests, "post", fake_post)
    monkeypatch.setattr(writer_module.logger, "warning", fake_warning)

    quest = Quest(1, "quest", "desc")
    writer = QuestWriter(board_channel="123", token="tok")

    writer.send_board_update(quest)

    assert warnings, "network failure should trigger warning log"

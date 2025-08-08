"""Tests for the quest FSM."""

from datetime import datetime, timedelta

import pytest

from deepthought.quest import QuestFSM, QuestState


# Helper to make deterministic times
BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


def test_valid_transition_sequence():
    fsm = QuestFSM(ttl_seconds=100, _last_refresh=BASE_TIME)
    fsm.transition(QuestState.TRACKED, now=BASE_TIME)
    fsm.transition(QuestState.ACTIVE, now=BASE_TIME)
    fsm.transition(QuestState.COMPLETED, now=BASE_TIME)
    assert fsm.state is QuestState.COMPLETED


def test_invalid_transition_raises():
    fsm = QuestFSM(ttl_seconds=100, _last_refresh=BASE_TIME)
    with pytest.raises(ValueError):
        fsm.transition(QuestState.ACTIVE, now=BASE_TIME)


def test_ttl_auto_prune():
    fsm = QuestFSM(ttl_seconds=10, _last_refresh=BASE_TIME)
    # move time forward beyond TTL
    later = BASE_TIME + timedelta(seconds=11)
    fsm.prune(now=later)
    assert fsm.state is QuestState.ABANDONED


def test_refresh_extends_ttl():
    fsm = QuestFSM(ttl_seconds=10, _last_refresh=BASE_TIME)
    half_time = BASE_TIME + timedelta(seconds=5)
    fsm.refresh(now=half_time)
    near_expiry = half_time + timedelta(seconds=9)
    fsm.prune(now=near_expiry)
    assert fsm.state is QuestState.PROPOSED
    # now exceed TTL after refresh
    expired = half_time + timedelta(seconds=11)
    fsm.prune(now=expired)
    assert fsm.state is QuestState.ABANDONED

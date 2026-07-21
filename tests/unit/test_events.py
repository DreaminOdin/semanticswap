"""ADR-012: Event-Bus - Emission, Subscription, History, Non-Blocking."""
import asyncio

import pytest

from semanticswap.events import EventBus


def test_emit_appends_history_with_ids():
    bus = EventBus()
    bus.emit("request", session="s1")
    bus.emit("monitor", tokens=42)
    assert [e["type"] for e in bus.history] == ["request", "monitor"]
    assert bus.history[0]["id"] < bus.history[1]["id"]
    assert bus.history[1]["tokens"] == 42


def test_history_is_bounded():
    bus = EventBus(history_size=5)
    for i in range(20):
        bus.emit("tick", n=i)
    assert len(bus.history) == 5
    assert bus.history[-1]["n"] == 19


@pytest.mark.asyncio
async def test_subscribers_receive_events():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    bus.emit("request")
    assert (await q1.get())["type"] == "request"
    assert (await q2.get())["type"] == "request"

    bus.unsubscribe(q1)
    bus.emit("monitor")
    assert q1.empty()
    assert (await q2.get())["type"] == "monitor"


@pytest.mark.asyncio
async def test_full_subscriber_queue_never_blocks():
    bus = EventBus()
    q = asyncio.Queue(maxsize=1)
    bus._subscribers.add(q)
    bus.emit("a")
    bus.emit("b")  # Queue voll -> Event verworfen, kein Fehler
    assert q.qsize() == 1

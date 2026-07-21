"""In-Process-Event-Bus für Live-Observability (ADR-012).

Kernpfade emittieren typisierte Events; Subscriber (z. B. der SSE-Stream der
GUI) konsumieren sie über asyncio-Queues. Emission ist fire-and-forget und
blockiert den Inferenz-Pfad nie: Volle Subscriber-Queues verlieren Events.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any


class EventBus:
    def __init__(self, history_size: int = 200):
        self._subscribers: set[asyncio.Queue] = set()
        self._history: deque[dict] = deque(maxlen=history_size)
        self._counter = 0

    def emit(self, event_type: str, **data: Any) -> dict:
        self._counter += 1
        event = {"id": self._counter, "ts": round(time.time(), 3),
                 "type": event_type, **data}
        self._history.append(event)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # langsamer Subscriber verliert Events, blockiert aber nie
        return event

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

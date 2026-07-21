"""Segmentierung des Chat-Verlaufs in logische Blöcke (PAD Phase 1, Schritt 1).

MVP-Strategie: ganze Messages werden greedy zu Segmenten von ~chunk_size Tokens
gruppiert; Messages werden nie zerschnitten (Schnitt-Präzision von Modus B).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..memory.store import StoredMessage
from ..tokens import TokenCounter


@dataclass
class RawSegment:
    start_idx: int
    end_idx: int  # inklusiv
    text: str


def render_message(msg: StoredMessage) -> str:
    return f"{msg.role.upper()}: {msg.content}"


def segment_messages(messages: list[StoredMessage], chunk_size: int,
                     counter: TokenCounter) -> list[RawSegment]:
    segments: list[RawSegment] = []
    buffer: list[StoredMessage] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            segments.append(RawSegment(
                start_idx=buffer[0].idx,
                end_idx=buffer[-1].idx,
                text="\n".join(render_message(m) for m in buffer),
            ))
            buffer, buffer_tokens = [], 0

    for msg in messages:
        msg_tokens = counter.count_text(msg.content) + 4
        if buffer and buffer_tokens + msg_tokens > chunk_size:
            flush()
        buffer.append(msg)
        buffer_tokens += msg_tokens
    flush()
    return segments

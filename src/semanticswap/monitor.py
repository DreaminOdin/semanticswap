"""Context Monitor: überwacht die Token-Zahl und triggert Kompression (PAD §1.2)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonitorResult:
    tokens: int
    ratio: float
    triggered_threshold: float | None


class ContextMonitor:
    def __init__(self, max_context_tokens: int, thresholds: list[float]):
        self.max_context_tokens = max_context_tokens
        self.thresholds = sorted(thresholds)

    def check(self, token_count: int) -> MonitorResult:
        ratio = token_count / self.max_context_tokens if self.max_context_tokens else 0.0
        triggered = None
        for t in self.thresholds:
            if ratio >= t:
                triggered = t
        return MonitorResult(tokens=token_count, ratio=ratio, triggered_threshold=triggered)

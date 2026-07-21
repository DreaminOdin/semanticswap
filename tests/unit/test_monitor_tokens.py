"""M2: Context Monitor und Token-Zählung."""
import threading
import time

from semanticswap import tokens
from semanticswap.monitor import ContextMonitor
from semanticswap.tokens import TokenCounter, plain_text


def test_plain_text_handles_content_parts():
    parts = [{"type": "text", "text": "Hallo"}, {"type": "image_url", "image_url": {}},
             {"type": "text", "text": "Welt"}]
    assert plain_text(parts) == "Hallo\nWelt"
    assert plain_text("direkt") == "direkt"
    assert plain_text(None) == ""


def test_heuristic_counter_is_deterministic():
    counter = TokenCounter(use_tiktoken=False)
    assert counter.count_text("") == 0
    assert counter.count_text("abcd" * 10) == 10


def test_tiktoken_load_never_blocks(monkeypatch):
    # Regression 2026-07-16 (Produktions-Freeze auf dem GX10): tiktoken lädt
    # sein Vokabular beim ersten Aufruf synchron aus dem Netz; ohne Egress
    # hing der Download und fror den kompletten Event-Loop ein. Das Laden
    # muss im Hintergrund passieren - bis dahin zählt die Heuristik.
    started = threading.Event()

    def hanging_load(self):
        started.set()
        time.sleep(30)  # simuliert hängenden Download

    monkeypatch.setattr(tokens.TokenCounter, "_load_encoding", hanging_load)
    counter = TokenCounter(use_tiktoken=True)
    t0 = time.monotonic()
    assert counter.count_text("abcd" * 10) == 10  # Heuristik, sofort
    assert time.monotonic() - t0 < 1.0
    assert started.wait(timeout=5)  # Laden wurde im Hintergrund angestoßen


def test_monitor_thresholds():
    monitor = ContextMonitor(max_context_tokens=100, thresholds=[0.5, 0.9])
    assert monitor.check(10).triggered_threshold is None
    assert monitor.check(50).triggered_threshold == 0.5
    assert monitor.check(95).triggered_threshold == 0.9
    assert monitor.check(95).ratio == 0.95

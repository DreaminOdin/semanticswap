"""Token-Zählung für den Context Monitor.

Nutzt tiktoken, wenn verfügbar (wird von litellm mitinstalliert); fällt sonst
auf eine Heuristik (~4 Zeichen pro Token) zurück, damit der Proxy auch offline
und in Tests deterministisch funktioniert.
"""
from __future__ import annotations

import threading
from typing import Any

_MSG_OVERHEAD = 4  # grobe Struktur-Token pro Message (role, Trennung)


def plain_text(content: Any) -> str:
    """Extrahiert reinen Text aus einem OpenAI-Message-Content (str oder Part-Liste)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content)


class TokenCounter:
    def __init__(self, model: str = "gpt-4o", use_tiktoken: bool = True):
        self.model = model
        self._use_tiktoken = use_tiktoken
        self._encoding = None
        self._load_started = False

    def _load_encoding(self):
        """Kann Netz-I/O machen (tiktoken lädt sein Vokabular beim ersten
        Gebrauch herunter) - darf deshalb NUR im Hintergrund-Thread laufen."""
        import tiktoken

        try:
            return tiktoken.encoding_for_model(self.model)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _get_encoding(self):
        if self._encoding is not None or not self._use_tiktoken:
            return self._encoding
        if not self._load_started:
            self._load_started = True

            def _run():
                try:
                    self._encoding = self._load_encoding()
                except Exception:
                    pass  # kein tiktoken/kein Netz -> Heuristik bleibt

            threading.Thread(target=_run, daemon=True,
                             name="tiktoken-load").start()
        # None, solange das Laden läuft oder scheiterte -> Heuristik
        return self._encoding

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        enc = self._get_encoding()
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict]) -> int:
        total = 2
        for msg in messages:
            total += _MSG_OVERHEAD
            total += self.count_text(plain_text(msg.get("content")))
        return total

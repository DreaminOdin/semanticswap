"""Auth v2 (docs/plan-auth-v2.md): Geräte-Cookies und Login-Bremse.

Alles mit Python-Bordmitteln (hmac/hashlib/secrets) — keine neuen
Abhängigkeiten. Jede Prüfung läuft auf dem GX10 selbst, weil die Funnel-URL
die Vercel-Edge umgeht.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

_DAY_SECONDS = 86400


def load_or_create_secret(path: Path) -> bytes:
    """Server-Secret für Cookie-Signaturen. Rotation (Datei löschen und
    Neustart) meldet alle Geräte ab."""
    if path.exists():
        return path.read_bytes()
    secret = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(secret)
    return secret


class DeviceCookieSigner:
    """HMAC-signierte Geräte-Tokens: base64url(payload) + '.' + Signatur.
    Payload: Rolle, Ablaufzeit, Nonce (macht Tokens einmalig)."""

    def __init__(self, secret: bytes):
        self._secret = secret

    def _sign(self, payload: bytes) -> str:
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def issue(self, role: str, days: int = 90) -> str:
        payload = json.dumps({
            "role": role,
            "exp": int(time.time()) + days * _DAY_SECONDS,
            "nonce": secrets.token_hex(8),
        }, sort_keys=True).encode("utf-8")
        encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
        return f"{encoded}.{self._sign(payload)}"

    def verify(self, token: str) -> str | None:
        """Gibt die Rolle zurück oder None (manipuliert/abgelaufen/fremd)."""
        try:
            encoded, signature = token.rsplit(".", 1)
            payload = base64.urlsafe_b64decode(encoded + "==")
        except Exception:
            return None
        if not hmac.compare_digest(self._sign(payload), signature):
            return None
        try:
            data = json.loads(payload)
        except Exception:
            return None
        if int(data.get("exp", 0)) <= time.time():
            return None
        role = data.get("role")
        return role if isinstance(role, str) and role else None


class LoginBrake:
    """Bremse gegen Passwort-Raten pro Quell-IP: nach `free_attempts`
    Fehlversuchen wächst eine Wartezeit exponentiell (gedeckelt)."""

    def __init__(self, free_attempts: int = 5, base_delay: float = 5.0,
                 max_delay: float = 300.0):
        self.free_attempts = free_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._failures: dict[str, tuple[int, float]] = {}  # ip -> (count, ts)

    def check(self, ip: str) -> tuple[bool, float]:
        """(erlaubt?, Restwartezeit in Sekunden)."""
        count, last = self._failures.get(ip, (0, 0.0))
        if count <= self.free_attempts:
            return True, 0.0
        delay = min(self.base_delay * 2 ** (count - self.free_attempts - 1),
                    self.max_delay)
        remaining = last + delay - time.monotonic()
        if remaining <= 0:
            return True, 0.0
        return False, remaining

    def register_failure(self, ip: str) -> None:
        count, _ = self._failures.get(ip, (0, 0.0))
        self._failures[ip] = (count + 1, time.monotonic())

    def register_success(self, ip: str) -> None:
        self._failures.pop(ip, None)


class AccessLog:
    """Durables Logbuch für Zugriffe außerhalb des Tailnets (Jan + Fremde).
    Anders als der In-Memory-Event-Bus überlebt es Neustarts (JSONL-Datei);
    ein Ringpuffer hält die jüngsten Einträge für die Admin-Ansicht bereit.
    Ohne `path` rein in-memory (Tests, :memory:-Betrieb)."""

    def __init__(self, path: Path | None = None, keep: int = 500,
                 max_bytes: int = 5_000_000):
        self.path = path
        self.max_bytes = max_bytes
        self._recent: deque[dict] = deque(maxlen=keep)
        if path and path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines()[-keep:]:
                    self._recent.append(json.loads(line))
            except Exception:
                pass

    def record(self, **fields) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 **fields}
        self._recent.append(entry)
        if not self.path:
            return
        try:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                self.path.replace(self.path.with_suffix(".jsonl.1"))  # 1 Rotation
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Logging darf den Request nie brechen

    def recent(self, n: int = 100) -> list[dict]:
        items = list(self._recent)[-n:]
        items.reverse()  # neueste zuerst
        return items

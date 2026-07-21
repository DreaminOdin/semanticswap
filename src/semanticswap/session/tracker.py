"""Session-Wiedererkennung über Prefix-Hash-Kette (ADR-003).

Die OpenAI-API ist stateless: Der Client schickt jedes Mal den vollen Verlauf.
Der Tracker ordnet eingehende Verläufe über eine Merkle-artige Hash-Kette der
richtigen Session zu. Regeln:

- Längster bekannter Prefix gewinnt; bei Mehrdeutigkeit die zuletzt aktive Session.
- Eine Session wird nur wiederverwendet, wenn der Client strikt anhängt
  (matched == gespeicherte Message-Zahl). Divergiert der Verlauf (Edit/Retry),
  wird ein Fork angelegt; der Archivstand wird übernommen, wenn er vor dem
  Divergenzpunkt liegt.
- Eine explizite Session-ID (Header x-session-id / OpenAI-Feld `user`) hat Vorrang.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from ..memory.store import Store

log = logging.getLogger(__name__)


def normalize_content(content: Any) -> str:
    """Kanonische Textform für das Hashing (Whitespace-/Struktur-Rauschen entfernen)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return " ".join(content.split())
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def message_hash(msg: dict) -> str:
    payload = f"{msg.get('role', '')}\x00{normalize_content(msg.get('content'))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chain_hashes(messages: list[dict]) -> list[str]:
    """Verkettete Prefix-Hashes: chain[i] identifiziert messages[0..i]."""
    chain: list[str] = []
    prev = ""
    for msg in messages:
        prev = hashlib.sha256((prev + message_hash(msg)).encode("ascii")).hexdigest()
        chain.append(prev)
    return chain


@dataclass
class Resolution:
    session_id: str
    matched_count: int   # wie viele eingehende Messages der Session bereits bekannt sind
    is_new: bool
    forked_from: str | None = None


class SessionTracker:
    def __init__(self, store: Store):
        self.store = store

    def resolve(self, messages: list[dict], explicit_id: str | None = None) -> Resolution:
        if explicit_id:
            return self._resolve_explicit(explicit_id, messages)

        chain = chain_hashes(messages)
        best_depth = 0
        best_candidates: list[tuple[str, int]] = []
        for depth, chash in enumerate(chain, start=1):
            candidates = self.store.lookup_chain(chash)
            if not candidates:
                break  # tiefere Prefixe können nicht mehr matchen
            best_depth = depth
            best_candidates = candidates

        if best_depth == 0:
            session = self.store.create_session()
            log.info("Neue Session %s (kein Prefix-Match)", session.id)
            return Resolution(session.id, 0, is_new=True)

        session_id = best_candidates[0][0]  # lookup_chain sortiert nach last_active
        session = self.store.get_session(session_id)
        stored = self.store.message_count(session_id)

        if session and best_depth == stored:
            self.store.touch_session(session_id)
            return Resolution(session_id, best_depth, is_new=False)

        # Divergenz: Client hat Historie editiert/gekürzt -> Fork (ADR-003 Regel 4).
        # Der Fork teilt den semantischen Speicher der Ursprungs-Session (ADR-009).
        fork = self.store.create_session(
            forked_from=session_id,
            memory_id=session.memory_id if session else None,
        )
        prefix_msgs = self.store.get_messages(session_id)[:best_depth]
        self.store.add_messages(fork.id, 0, [m.raw for m in prefix_msgs])
        self.store.add_chain_links(fork.id, [(h, i + 1) for i, h in enumerate(chain[:best_depth])])
        if session and 0 < session.archived_upto <= best_depth:
            self.store.set_archive(fork.id, session.archived_upto, session.archive_prompt)
        log.warning(
            "Session %s divergiert bei Message %d (gespeichert: %d) -> Fork %s",
            session_id, best_depth, stored, fork.id,
        )
        return Resolution(fork.id, best_depth, is_new=True, forked_from=session_id)

    def _resolve_explicit(self, session_id: str, messages: list[dict]) -> Resolution:
        session = self.store.get_session(session_id)
        if session is None:
            self.store.create_session(session_id=session_id)
            return Resolution(session_id, 0, is_new=True)
        chain = chain_hashes(messages)
        matched = 0
        for depth, chash in enumerate(chain, start=1):
            if any(sid == session_id for sid, _ in self.store.lookup_chain(chash)):
                matched = depth
            else:
                break
        self.store.touch_session(session_id)
        return Resolution(session_id, matched, is_new=False)

    def record(self, session_id: str, messages: list[dict]) -> None:
        """Persistiert den finalen Verlauf eines Turns (inkl. Assistant-Antwort)."""
        stored = self.store.message_count(session_id)
        if len(messages) < stored:
            log.warning("record(): eingehender Verlauf kürzer als gespeichert (%d < %d)",
                        len(messages), stored)
            return
        chain = chain_hashes(messages)
        if len(messages) > stored:
            self.store.add_messages(session_id, stored, messages[stored:])
        self.store.add_chain_links(
            session_id, [(h, i + 1) for i, h in enumerate(chain)]
        )
        self.store.touch_session(session_id)

"""Semantic Memory Layer: SQLite-Persistenz (Sessions, Hash-Kette, Messages,
Segmente, Graph-Tripel, Embeddings). Siehe ADR-004.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    archived_upto INTEGER NOT NULL DEFAULT 0,
    archive_prompt TEXT NOT NULL DEFAULT '',
    compressing INTEGER NOT NULL DEFAULT 0,
    forked_from TEXT,
    memory_id TEXT
);
CREATE TABLE IF NOT EXISTS chain_links (
    chain_hash TEXT NOT NULL,
    session_id TEXT NOT NULL,
    msg_index INTEGER NOT NULL,
    PRIMARY KEY (chain_hash, session_id)
);
CREATE INDEX IF NOT EXISTS idx_chain_hash ON chain_links(chain_hash);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    start_idx INTEGER NOT NULL,
    end_idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'high',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    segment_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    content, segment_id UNINDEXED, session_id UNINDEXED,
    tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS profiles (
    memory_id TEXT PRIMARY KEY,
    text TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    id: str
    created_at: str
    last_active: str
    archived_upto: int
    archive_prompt: str
    compressing: bool
    forked_from: str | None
    memory_id: str  # geteilter Speicherraum; Forks erben ihn (ADR-009)


@dataclass
class StoredMessage:
    idx: int
    role: str
    content: str
    raw: dict


@dataclass
class Segment:
    id: str
    session_id: str
    start_idx: int
    end_idx: int
    text: str
    summary: str = ""
    priority: str = "high"  # "high" | "low" (ADR-011)


class Store:
    def __init__(self, db_path: str | Path = ":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Mini-Migrationen für Datenbanken aus älteren Ständen
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)")}
            if "memory_id" not in cols:  # ADR-009
                self._conn.execute("ALTER TABLE sessions ADD COLUMN memory_id TEXT")
            seg_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(segments)")}
            if "priority" not in seg_cols:  # ADR-011
                self._conn.execute(
                    "ALTER TABLE segments ADD COLUMN priority TEXT NOT NULL DEFAULT 'high'")
            # FTS-Backfill für Datenbanken aus Zeiten vor der Hybrid-Suche
            self._conn.execute(
                "INSERT INTO segments_fts (content, segment_id, session_id) "
                "SELECT text || ' ' || summary, id, session_id FROM segments "
                "WHERE id NOT IN (SELECT segment_id FROM segments_fts)")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- Sessions -----------------------------------------------------------

    def create_session(self, session_id: str | None = None,
                       forked_from: str | None = None,
                       memory_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:16]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, created_at, last_active, forked_from, "
                "memory_id) VALUES (?, ?, ?, ?, ?)",
                (sid, now, now, forked_from, memory_id or sid),
            )
            self._conn.commit()
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> Session | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"], created_at=row["created_at"], last_active=row["last_active"],
            archived_upto=row["archived_upto"], archive_prompt=row["archive_prompt"],
            compressing=bool(row["compressing"]), forked_from=row["forked_from"],
            memory_id=row["memory_id"] or row["id"],
        )

    def list_sessions(self) -> list[Session]:
        rows = self._conn.execute(
            "SELECT id FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        return [self.get_session(r["id"]) for r in rows]  # type: ignore[misc]

    def touch_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (_now(), session_id)
            )
            self._conn.commit()

    def set_archive(self, session_id: str, archived_upto: int, archive_prompt: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET archived_upto = ?, archive_prompt = ? WHERE id = ?",
                (archived_upto, archive_prompt, session_id),
            )
            self._conn.commit()

    def try_mark_compressing(self, session_id: str) -> bool:
        """Atomarer Statuswechsel; verhindert doppelte Kompressions-Jobs."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET compressing = 1 WHERE id = ? AND compressing = 0",
                (session_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def clear_compressing(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET compressing = 0 WHERE id = ?", (session_id,)
            )
            self._conn.commit()

    # --- Hash-Kette (ADR-003) ------------------------------------------------

    def add_chain_links(self, session_id: str, links: list[tuple[str, int]]) -> None:
        """links: Liste von (chain_hash, msg_index/Prefix-Länge)."""
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO chain_links (chain_hash, session_id, msg_index) "
                "VALUES (?, ?, ?)",
                [(h, session_id, i) for h, i in links],
            )
            self._conn.commit()

    def lookup_chain(self, chain_hash: str) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT c.session_id, c.msg_index FROM chain_links c "
            "JOIN sessions s ON s.id = c.session_id "
            "WHERE c.chain_hash = ? ORDER BY s.last_active DESC",
            (chain_hash,),
        ).fetchall()
        return [(r["session_id"], r["msg_index"]) for r in rows]

    # --- Messages -------------------------------------------------------------

    def add_messages(self, session_id: str, start_idx: int, messages: list[dict]) -> None:
        rows = []
        for offset, msg in enumerate(messages):
            from .. import tokens as _t
            rows.append((
                session_id, start_idx + offset, msg.get("role", "user"),
                _t.plain_text(msg.get("content")),
                json.dumps(msg, ensure_ascii=False, sort_keys=True),
            ))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO messages (session_id, idx, role, content, raw_json) "
                "VALUES (?, ?, ?, ?, ?)", rows,
            )
            self._conn.commit()

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        rows = self._conn.execute(
            "SELECT idx, role, content, raw_json FROM messages "
            "WHERE session_id = ? ORDER BY idx", (session_id,),
        ).fetchall()
        return [
            StoredMessage(idx=r["idx"], role=r["role"], content=r["content"],
                          raw=json.loads(r["raw_json"]))
            for r in rows
        ]

    def message_count(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["n"])

    # --- Segmente / Tripel / Embeddings ---------------------------------------

    def add_segment(self, segment: Segment) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments (id, session_id, start_idx, end_idx, "
                "text, summary, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (segment.id, segment.session_id, segment.start_idx, segment.end_idx,
                 segment.text, segment.summary, segment.priority, _now()),
            )
            self._conn.execute(
                "DELETE FROM segments_fts WHERE segment_id = ?", (segment.id,))
            self._conn.execute(
                "INSERT INTO segments_fts (content, segment_id, session_id) "
                "VALUES (?, ?, ?)",
                (f"{segment.text} {segment.summary}", segment.id,
                 segment.session_id),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_segment(r) -> Segment:
        return Segment(id=r["id"], session_id=r["session_id"],
                       start_idx=r["start_idx"], end_idx=r["end_idx"],
                       text=r["text"], summary=r["summary"], priority=r["priority"])

    def get_segments(self, session_id: str) -> list[Segment]:
        rows = self._conn.execute(
            "SELECT id, session_id, start_idx, end_idx, text, summary, priority "
            "FROM segments WHERE session_id = ? ORDER BY start_idx", (session_id,),
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_segment(self, segment_id: str) -> Segment | None:
        r = self._conn.execute(
            "SELECT id, session_id, start_idx, end_idx, text, summary, priority "
            "FROM segments WHERE id = ?", (segment_id,),
        ).fetchone()
        return None if r is None else self._row_to_segment(r)

    def keyword_search(self, query: str, top_k: int = 5,
                       session_id: str | None = None) -> list[tuple[str, float]]:
        """FTS5-Stichwortsuche (BM25) über Segment-Texte + Summaries.
        Ergänzt die Vektorsuche um exakte Treffer (Namen, Zahlen, Codes) —
        Hybrid-Retrieval, Diagnose aus LongMemEval-Pilot #1."""
        terms = re.findall(r"[\wäöüßÄÖÜ-]{2,}", query)[:12]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = ("SELECT segment_id, bm25(segments_fts) AS rank "
               "FROM segments_fts WHERE segments_fts MATCH ?")
        params: list[Any] = [match]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # exotische Query-Syntax -> kein Treffer
            return []
        return [(r["segment_id"], float(r["rank"])) for r in rows]

    def add_triples(self, session_id: str, segment_id: str,
                    triples: list[tuple[str, str, str]]) -> None:
        with self._lock:
            # Idempotente Nach-Archivierung: alte Tripel des Segments ersetzen
            self._conn.execute("DELETE FROM triples WHERE segment_id = ?",
                               (segment_id,))
            self._conn.executemany(
                "INSERT INTO triples (session_id, segment_id, subject, predicate, object) "
                "VALUES (?, ?, ?, ?, ?)",
                [(session_id, segment_id, s, p, o) for s, p, o in triples],
            )
            self._conn.commit()

    def expand_by_graph(self, seed_segment_ids: list[str], memory_id: str,
                        limit: int = 3) -> list[tuple[str, int]]:
        """Graph-Expansion (Iteration D): Nachbar-Segmente, die mit den Seed-
        Segmenten Entitäten (Subjekt/Objekt) teilen — innerhalb desselben
        Speicherraums. Rückgabe nach Zahl gemeinsamer Entitäten sortiert.
        Verbindet über Sessions hinweg, was der Graph bereits verknüpft."""
        if not seed_segment_ids:
            return []
        seeds = set(seed_segment_ids)
        ph = ",".join("?" * len(seed_segment_ids))
        entity_rows = self._conn.execute(
            f"SELECT subject, object FROM triples WHERE segment_id IN ({ph})",
            seed_segment_ids).fetchall()
        entities = set()
        for r in entity_rows:
            entities.add(r["subject"])
            entities.add(r["object"])
        if not entities:
            return []
        eph = ",".join("?" * len(entities))
        params: list[Any] = [memory_id, *entities, *entities]
        rows = self._conn.execute(
            f"SELECT segment_id, COUNT(DISTINCT COALESCE(subject, '') || '|' || "
            f"COALESCE(object, '')) AS shared FROM triples "
            f"WHERE session_id = ? AND (subject IN ({eph}) OR object IN ({eph})) "
            f"GROUP BY segment_id ORDER BY shared DESC",
            params).fetchall()
        out = [(r["segment_id"], int(r["shared"])) for r in rows
               if r["segment_id"] not in seeds]
        return out[:limit]

    def get_triples(self, session_id: str) -> list[tuple[str, str, str]]:
        rows = self._conn.execute(
            "SELECT subject, predicate, object FROM triples WHERE session_id = ? "
            "ORDER BY id", (session_id,),
        ).fetchall()
        return [(r["subject"], r["predicate"], r["object"]) for r in rows]

    def get_triples_with_recency(
            self, session_id: str) -> list[tuple[str, str, str, int]]:
        """Wie get_triples, aber mit der Recency (start_idx des Quell-Segments)
        als viertem Element — Grundlage der temporalen Verdrängung (Iteration C).
        Tripel ohne auffindbares Segment bekommen Recency -1 (gelten als alt)."""
        rows = self._conn.execute(
            "SELECT t.subject, t.predicate, t.object, "
            "COALESCE(s.start_idx, -1) AS recency FROM triples t "
            "LEFT JOIN segments s ON s.id = t.segment_id "
            "WHERE t.session_id = ? ORDER BY t.id", (session_id,),
        ).fetchall()
        return [(r["subject"], r["predicate"], r["object"], int(r["recency"]))
                for r in rows]

    def set_profile(self, memory_id: str, text: str) -> None:
        """Stehendes Nutzerprofil pro Speicherraum (Iteration B)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO profiles (memory_id, text) VALUES (?, ?)",
                (memory_id, text))
            self._conn.commit()

    def get_profile(self, memory_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM profiles WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return row["text"] if row else None

    def set_embedding(self, segment_id: str, vector: bytes, dim: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (segment_id, vector, dim) "
                "VALUES (?, ?, ?)", (segment_id, vector, dim),
            )
            self._conn.commit()

    def all_embeddings(self, session_id: str | None = None) -> list[tuple[str, bytes, int]]:
        if session_id:
            rows = self._conn.execute(
                "SELECT e.segment_id, e.vector, e.dim FROM embeddings e "
                "JOIN segments s ON s.id = e.segment_id WHERE s.session_id = ?",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT segment_id, vector, dim FROM embeddings"
            ).fetchall()
        return [(r["segment_id"], r["vector"], r["dim"]) for r in rows]

    # --- Statistiken (Admin-API, ADR-006) --------------------------------------

    def stats(self) -> dict[str, Any]:
        def _count(table: str) -> int:
            return int(self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

        seg = self._conn.execute(
            "SELECT COALESCE(SUM(LENGTH(text)), 0) AS orig, "
            "COALESCE(SUM(LENGTH(summary)), 0) AS comp, "
            "COALESCE(SUM(priority = 'low'), 0) AS low FROM segments").fetchone()
        prompt = self._conn.execute(
            "SELECT COALESCE(SUM(LENGTH(archive_prompt)), 0) AS n FROM sessions"
        ).fetchone()
        return {
            "sessions": _count("sessions"),
            "messages": _count("messages"),
            "segments": _count("segments"),
            "triples": _count("triples"),
            "embeddings": _count("embeddings"),
            # Auswertung (Zeichenlängen als Token-Proxy, GUI-KPIs):
            "archived_chars": int(seg["orig"]),
            "summary_chars": int(seg["comp"]),
            "prompt_chars": int(prompt["n"]),
            "low_priority_segments": int(seg["low"]),
        }

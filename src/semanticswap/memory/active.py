"""Active Memory Retrieval (ADR-010): LLM-gesteuerter Swap-In per Function Calling.

Das Haupt-LLM erhält das Tool `retrieve_archived_memory` und kann archivierte
Original-Segmente selbst anfordern. Der Tool-Zyklus läuft vollständig im Proxy;
der Client sieht nur die finale Antwort.
"""
from __future__ import annotations

import json
import logging

from ..config import AppConfig
from ..memory.retrieval import Retriever
from ..memory.store import Store

log = logging.getLogger(__name__)

TOOL_NAME = "retrieve_archived_memory"

_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Retrieve original transcripts from the compressed conversation "
            "archive. Use this when the ARCHIVE summary mentions a topic or a "
            "#segment reference whose details you need to answer precisely. "
            "Provide a search query and/or an exact segment_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Semantic search query for the archive"},
                "segment_id": {"type": "string",
                               "description": "Exact segment reference, e.g. seg_ab12cd34_0001_0004"},
            },
        },
    },
}

MAX_TOOL_ROUNDS = 2


class ActiveRetrieval:
    def __init__(self, store: Store, retriever: Retriever, cfg: AppConfig):
        self.store = store
        self.retriever = retriever
        self.cfg = cfg

    def should_enable(self, session, body: dict) -> bool:
        """Tool nur injizieren, wenn es keine Kollision mit Harness-Tools geben
        kann (ADR-010): kein Streaming, keine Client-Tools, Archiv vorhanden."""
        return bool(
            self.cfg.retrieval.active_tool
            and self.cfg.retrieval.enabled
            and session is not None
            and session.archived_upto > 0
            and session.archive_prompt
            and not body.get("stream")
            and not body.get("tools")
            and not body.get("functions")  # Legacy-Function-Calling
        )

    @staticmethod
    def tool_definition() -> dict:
        return _TOOL_DEFINITION

    @staticmethod
    def own_tool_calls(message: dict) -> list[dict]:
        calls = message.get("tool_calls") or []
        return [c for c in calls
                if (c.get("function") or {}).get("name") == TOOL_NAME]

    async def resolve(self, session_id: str, tool_call: dict) -> str:
        """Führt einen Tool-Aufruf aus; liefert immer einen Text (best effort)."""
        try:
            args = json.loads((tool_call.get("function") or {}).get("arguments") or "{}")
        except (json.JSONDecodeError, ValueError):
            args = {}
        session = self.store.get_session(session_id)
        if session is None:
            return "Archive unavailable."

        segment_id = (args.get("segment_id") or "").strip().lstrip("#")
        if segment_id:
            segment = self.store.get_segment(segment_id)
            if segment is not None and segment.session_id == session.memory_id:
                budget_chars = self.cfg.retrieval.max_injection_tokens * 4
                return (f"--- #{segment.id} (messages {segment.start_idx}-"
                        f"{segment.end_idx}) ---\n{segment.text[:budget_chars]}")
            log.info("Active Retrieval: unbekannte segment_id %r", segment_id)

        query = (args.get("query") or "").strip()
        if query:
            snippets = await self.retriever.snippets_for_query(session_id, query)
            if snippets:
                return snippets
        return ("No matching archived content found. Answer from the ARCHIVE "
                "summary or state that the detail is not available.")

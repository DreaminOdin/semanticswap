"""Swap-Out-Pipeline (Modus B, Post-Response Batch) und Hintergrund-Queue.

Ablauf pro Session (PAD Phase 1):
  Segmentierung -> parallele Worker (Map) -> Synthese des ARCHIVE-Prompts (Reduce)
  -> Persistenz (Segmente, Tripel, Embeddings) -> Zeiger `archived_upto` setzen.

Der aktive Client-Verlauf wird dabei nie verändert - das Pruning ist virtuell
(ADR-003): Beim nächsten Request ersetzt das Gateway die archivierten Messages
durch den ARCHIVE-Prompt.
"""
from __future__ import annotations

import asyncio
import logging

from ..config import AppConfig
from ..events import EventBus
from ..llm import LLMClient
from ..memory.store import Segment, Store
from ..memory.vector import NaiveVectorStore
from ..prompts import build_archive_prompt
from ..tokens import TokenCounter
from .segmenter import segment_messages
from .workers import ExtractionWorkers

log = logging.getLogger(__name__)


class CompressionPipeline:
    def __init__(self, store: Store, workers: ExtractionWorkers,
                 counter: TokenCounter, cfg: AppConfig,
                 llm: LLMClient | None = None, bus: EventBus | None = None):
        self.store = store
        self.workers = workers
        self.counter = counter
        self.cfg = cfg
        self.llm = llm
        self.bus = bus
        self.vectors = NaiveVectorStore(store)

    def _emit(self, event_type: str, **data) -> None:
        if self.bus is not None:
            self.bus.emit(event_type, **data)

    def render_archive_prompt(self, memory_id: str) -> str:
        """Baut den ARCHIVE-Prompt aus dem gespeicherten Speicher neu. Zentral,
        damit Ingestion und (für Cache-Experimente) ein späterer Neuaufbau
        exakt dieselbe Logik nutzen — inkl. temporaler Verdrängung (Iteration C)
        und Nutzerprofil (Iteration B)."""
        segments = self.store.get_segments(memory_id)
        if self.cfg.compression.temporal_supersede:
            triples = self.store.get_triples_with_recency(memory_id)
        else:
            triples = self.store.get_triples(memory_id)
        profile = (self.store.get_profile(memory_id)
                   if self.cfg.compression.user_profile else None)
        return build_archive_prompt(
            segments, triples,
            low_priority_visible=self.cfg.compression.low_priority_visible,
            temporal_supersede=self.cfg.compression.temporal_supersede,
            user_profile=profile,
            entity_resolution=self.cfg.compression.entity_resolution)

    async def build_and_store_profile(self, memory_id: str) -> str:
        """Iteration B: Profil aus den gespeicherten Summaries destillieren und
        ablegen. Getrennt vom Rendern, damit der Eval es auf der Cache-Kopie
        nachziehen kann (ohne Re-Ingestion)."""
        segments = self.store.get_segments(memory_id)
        summaries = [s.summary for s in segments if (s.summary or "").strip()]
        profile = await self.workers.build_profile(summaries)
        if profile.strip():
            self.store.set_profile(memory_id, profile)
        return profile

    async def compress_session(self, session_id: str) -> int:
        """Komprimiert alle noch nicht archivierten Messages (bis auf die letzten
        keep_recent_messages). Gibt die Zahl neu erzeugter Segmente zurück."""
        session = self.store.get_session(session_id)
        if session is None:
            return 0
        messages = self.store.get_messages(session_id)
        if not messages:
            return 0

        start = session.archived_upto
        if start == 0 and messages and messages[0].role == "system":
            start = 1  # System-Prompt des Clients bleibt immer erhalten
        cutoff = len(messages) - self.cfg.compression.keep_recent_messages
        if cutoff <= start:
            return 0

        raw_segments = segment_messages(
            messages[start:cutoff], self.cfg.storage.chunk_size, self.counter
        )
        self._emit("compression_start", session=session_id,
                   segments=len(raw_segments))

        async def process(raw):
            label = f"msg {raw.start_idx}-{raw.end_idx}"
            self._emit("subagent_start", session=session_id, segment=label)
            result = await self.workers.process_segment(raw.text)
            self._emit("subagent_done", session=session_id, segment=label,
                       triples=len(result.triples), priority=result.priority)
            return result

        results = await asyncio.gather(*(process(seg) for seg in raw_segments))

        # Speicher-Artefakte liegen im geteilten Speicherraum (memory_id, ADR-009)
        memory_id = session.memory_id
        new_segments: list[Segment] = []
        for raw, result in zip(raw_segments, results):
            segment = Segment(
                id=f"seg_{memory_id[:8]}_{raw.start_idx:04d}_{raw.end_idx:04d}",
                session_id=memory_id,
                start_idx=raw.start_idx,
                end_idx=raw.end_idx,
                text=raw.text,
                summary=result.summary,
                priority=result.priority,
            )
            self.store.add_segment(segment)
            if result.triples:
                self.store.add_triples(memory_id, segment.id, result.triples)
            new_segments.append(segment)

        await self._embed_segments(new_segments)

        # Iteration B: Nutzerprofil vor dem Rendern aktualisieren
        if self.cfg.compression.user_profile:
            await self.build_and_store_profile(memory_id)

        # Reduce-Schritt: konsolidierter ARCHIVE-Prompt über ALLE Segmente des Speichers
        archive_prompt = self.render_archive_prompt(memory_id)
        self.store.set_archive(session_id, cutoff, archive_prompt)
        self._emit("archive_updated", session=session_id,
                   new_segments=len(new_segments), archived_upto=cutoff,
                   archive_tokens=self.counter.count_text(archive_prompt))
        log.info("Session %s: %d Segment(e) archiviert, archived_upto=%d",
                 session_id, len(new_segments), cutoff)
        return len(new_segments)

    async def _embed_segments(self, segments: list[Segment]) -> None:
        if not (self.cfg.embedding.enabled and self.llm and segments):
            return
        try:
            texts = [seg.summary or seg.text for seg in segments]
            extra = {}
            if self.cfg.embedding.api_base:
                extra["api_base"] = self.cfg.embedding.api_base
            if self.cfg.embedding.api_key:
                extra["api_key"] = self.cfg.embedding.api_key
            vectors = await self.llm.embed(self.cfg.embedding.model, texts, **extra)
            for seg, vec in zip(segments, vectors):
                self.vectors.add(seg.id, vec)
        except Exception:
            log.exception("Embedding fehlgeschlagen - Segmente bleiben ohne Vektor")


class BackgroundCompressor:
    """Asynchrone Job-Queue (PAD §5.3): Kompression läuft im Hintergrund und
    blockiert den API-Thread nie. Doppel-Enqueues werden über das
    `compressing`-Flag der Session verhindert."""

    def __init__(self, pipeline: CompressionPipeline):
        self.pipeline = pipeline
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    def enqueue(self, session_id: str) -> bool:
        if not self.pipeline.store.try_mark_compressing(session_id):
            return False
        self._ensure_started()
        self.queue.put_nowait(session_id)
        return True

    async def _run(self) -> None:
        while True:
            session_id = await self.queue.get()
            try:
                await self.pipeline.compress_session(session_id)
            except Exception:
                log.exception("Kompression für Session %s fehlgeschlagen", session_id)
            finally:
                self.pipeline.store.clear_compressing(session_id)
                self.queue.task_done()

    async def drain(self) -> None:
        """Wartet, bis alle anstehenden Jobs verarbeitet sind (Tests/Shutdown)."""
        await self.queue.join()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

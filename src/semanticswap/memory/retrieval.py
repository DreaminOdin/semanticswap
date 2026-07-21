"""Swap-In v1 (ADR-008): bedarfsgerechte Rekonstruktion (PAD Phase 2).

Die letzte User-Message wird als Query embedded; passende archivierte
Original-Segmente werden - begrenzt durch ein Token-Budget - als temporäre
System-Message für den aktuellen Upstream-Request aufbereitet. Best effort:
jeder Fehler degradiert zu "keine Injection".
"""
from __future__ import annotations

import logging

import re

from ..config import AppConfig
from ..events import EventBus
from ..llm import LLMClient
from ..prompts import DECOMPOSE_PROMPT, RERANK_PROMPT, build_retrieval_prompt
from ..tokens import TokenCounter, plain_text
from .store import Store
from .vector import NaiveVectorStore

log = logging.getLogger(__name__)

_MIN_USEFUL_TOKENS = 50
# Cross-Encoder (ms-marco-MiniLM) verkraftet ~512 Token ≈ 2000 Zeichen. Zu
# starkes Kürzen (früher 500) zeigt dem Ranker nur den Segment-Anfang und
# verfälscht das Scoring. Fürs LLM-Listwise-Backend bleibt es ein Kostenlimit.
_RERANK_SNIPPET_CHARS = 1800


def _parse_subqueries(raw: str) -> list[str]:
    """Zerlegt die LLM-Antwort in Teilfragen (nummerierte oder Listen-Zeilen)."""
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(\d+[.)]\s*|[-*]\s*)", "", line).strip()
        if line:
            out.append(line)
    return out


def _parse_rank_order(raw: str, n: int) -> list[int] | None:
    """Wandelt eine LLM-Rangantwort ("2, 1, 3") in 0-basierte Indizes um.
    Out-of-range/Duplikate werden verworfen; unparsebar -> None (Fallback)."""
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    order: list[int] = []
    seen = set()
    for num in nums:
        idx = num - 1  # Prompt ist 1-basiert
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order or None


def _rrf_merge(ranked_lists: list[list[str]], top_k: int,
               k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: verschmilzt Rank-Listen verschiedener
    Suchverfahren; Treffer in mehreren Listen steigen nach oben."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ordered[:top_k]]


class Retriever:
    def __init__(self, store: Store, vectors: NaiveVectorStore, llm: LLMClient,
                 counter: TokenCounter, cfg: AppConfig,
                 bus: EventBus | None = None, reranker=None):
        self.store = store
        self.vectors = vectors
        self._reranker = reranker  # injizierbar (Tests); sonst lazy geladen
        self.llm = llm
        self.counter = counter
        self.cfg = cfg
        self.bus = bus

    async def _rerank(self, query: str, hits: list[tuple[str, float]],
                      session_id: str) -> list[tuple[str, float]]:
        """Re-Ranking der Kandidaten. Backend 'cross_encoder' (ONNX, verlässlich)
        oder 'llm' (Listwise, schwach bei kleinen Modellen). Best effort:
        Fehler -> ursprüngliche Reihenfolge."""
        candidates = []
        for seg_id, score in hits:
            seg = self.store.get_segment(seg_id)
            if seg is not None:
                candidates.append((seg_id, score, seg.text))
        if len(candidates) < 2:
            return hits

        if self.cfg.retrieval.rerank_backend == "cross_encoder":
            return await self._rerank_cross_encoder(query, candidates, session_id)

        snippets = "\n".join(
            f"[{i + 1}] {text[:_RERANK_SNIPPET_CHARS]}"
            for i, (_, _, text) in enumerate(candidates))
        model = self.cfg.retrieval.rerank_model or self.cfg.sub_agents.tasks.summarization
        try:
            extra = {}
            if self.cfg.sub_agents.api_base:
                extra["api_base"] = self.cfg.sub_agents.api_base
            if self.cfg.sub_agents.api_key:
                extra["api_key"] = self.cfg.sub_agents.api_key
            resp = await self.llm.complete(
                model, [{"role": "user", "content": RERANK_PROMPT.format(
                    query=query, snippets=snippets)}],
                temperature=0, **extra)
            raw = resp["choices"][0]["message"].get("content") or ""
        except Exception:
            log.exception("Re-Ranker fehlgeschlagen - Reihenfolge unverändert")
            return hits
        order = _parse_rank_order(raw, len(candidates))
        if order is None:
            return hits
        # Nicht genannte Kandidaten hinten anhängen (Reihenfolge stabil)
        ranked = [candidates[i] for i in order]
        ranked += [c for j, c in enumerate(candidates) if j not in set(order)]
        if self.bus is not None:
            self.bus.emit("rerank", session=session_id,
                          candidates=len(candidates), model=model)
        return [(seg_id, score) for seg_id, score, _ in ranked]

    async def _decompose(self, query: str, session_id: str) -> list[str]:
        """Prio 3: Multi-Hop-Frage in Teilfragen zerlegen. Best effort:
        Fehler/leere Antwort -> [query] (unverändert)."""
        model = self.cfg.retrieval.rerank_model or self.cfg.sub_agents.tasks.summarization
        try:
            extra = {}
            if self.cfg.sub_agents.api_base:
                extra["api_base"] = self.cfg.sub_agents.api_base
            if self.cfg.sub_agents.api_key:
                extra["api_key"] = self.cfg.sub_agents.api_key
            resp = await self.llm.complete(
                model, [{"role": "user", "content": DECOMPOSE_PROMPT.format(
                    query=query, maxn=self.cfg.retrieval.query_decompose_max)}],
                temperature=0, **extra)
            subs = _parse_subqueries(resp["choices"][0]["message"].get("content") or "")
        except Exception:
            log.exception("Query-Decomposition fehlgeschlagen")
            return [query]
        subs = subs[:self.cfg.retrieval.query_decompose_max] or [query]
        if len(subs) > 1 and self.bus is not None:
            self.bus.emit("query_decompose", session=session_id, parts=len(subs))
        return subs

    async def _candidate_ids(self, query: str, session, first_stage_k: int,
                             session_id: str) -> tuple[list[str], dict[str, float]]:
        """Kandidaten-IDs für EINE (Teil-)Frage: Hybrid (Vektor+FTS+RRF) oder
        reine Vektorsuche. Gibt geordnete IDs + Vektor-Scores zurück."""
        query_vec = None
        if self.cfg.embedding.enabled:
            try:
                extra = {}
                if self.cfg.embedding.api_base:
                    extra["api_base"] = self.cfg.embedding.api_base
                if self.cfg.embedding.api_key:
                    extra["api_key"] = self.cfg.embedding.api_key
                vectors = await self.llm.embed(self.cfg.embedding.model, [query], **extra)
                query_vec = vectors[0]
            except Exception:
                if not self.cfg.retrieval.hybrid:
                    log.exception("Query-Embedding fehlgeschlagen - übersprungen")
                    return [], {}
                log.warning("Query-Embedding fehlgeschlagen - nur Volltextsuche")

        vec_scores: dict[str, float] = {}
        if self.cfg.retrieval.hybrid:
            vec_ids: list[str] = []
            if query_vec is not None:
                for seg_id, score in self.vectors.search(
                        query_vec, first_stage_k * 2, session.memory_id):
                    vec_ids.append(seg_id)
                    vec_scores[seg_id] = score
            kw_ids = [seg_id for seg_id, _ in self.store.keyword_search(
                query, first_stage_k * 2, session.memory_id)]
            ids = _rrf_merge([vec_ids, kw_ids], first_stage_k)
            if self.bus is not None:
                self.bus.emit("retrieval_search", session=session_id,
                              mode="hybrid", vec=len(vec_ids), kw=len(kw_ids),
                              fused=len(ids))
        else:
            scored = [(seg_id, score) for seg_id, score in self.vectors.search(
                query_vec, first_stage_k, session.memory_id)
                if query_vec is not None and score >= self.cfg.retrieval.min_score]
            vec_scores = dict(scored)
            ids = [seg_id for seg_id, _ in scored]
            if self.bus is not None:
                self.bus.emit("retrieval_search", session=session_id,
                              mode="vector", vec=len(ids), kw=0, fused=len(ids))
        return ids, vec_scores

    async def _rerank_cross_encoder(self, query: str, candidates: list,
                                    session_id: str) -> list[tuple[str, float]]:
        """ONNX-Cross-Encoder: (Query, Passage)-Paare direkt bepunkten,
        nach Score absteigend sortieren. Scoring läuft synchron (onnxruntime)
        in einem Thread, damit der Event-Loop frei bleibt."""
        import asyncio

        reranker = self._reranker
        if reranker is None:
            try:
                from .reranker import get_reranker
                reranker = get_reranker(self.cfg.retrieval.rerank_model)
                self._reranker = reranker
            except Exception:
                log.exception("Cross-Encoder nicht verfügbar - Reihenfolge unverändert")
                return [(sid, sc) for sid, sc, _ in candidates]
        docs = [text[:_RERANK_SNIPPET_CHARS] for _, _, text in candidates]
        try:
            scores = await asyncio.to_thread(reranker.score, query, docs)
        except Exception:
            log.exception("Cross-Encoder-Scoring fehlgeschlagen")
            return [(sid, sc) for sid, sc, _ in candidates]
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1],
                        reverse=True)
        if self.bus is not None:
            self.bus.emit("rerank", session=session_id,
                          candidates=len(candidates), model="cross_encoder")
        return [(sid, sc) for (sid, sc, _), _ in ranked]

    @staticmethod
    def _query_text(messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return plain_text(msg.get("content")).strip()
        return ""

    async def build_injection(self, session_id: str,
                              messages: list[dict]) -> dict | None:
        """Liefert eine System-Message mit Original-Snippets oder None."""
        if not (self.cfg.retrieval.enabled and self.cfg.embedding.enabled):
            return None
        query = self._query_text(messages)
        content = await self.snippets_for_query(session_id, query)
        if content is None:
            return None
        return {"role": "system", "content": content}

    async def snippets_for_query(self, session_id: str,
                                 query: str) -> str | None:
        """Kern des Swap-In: Vektorsuche + Token-Budget, gerendert als Text.
        Wird vom passiven Swap-In (ADR-008) und vom Active-Retrieval-Tool
        (ADR-010) gemeinsam genutzt. Best effort: Fehler ergeben None."""
        if not self.cfg.embedding.enabled:
            return None
        session = self.store.get_session(session_id)
        if session is None:
            return None
        if not query:
            return None
        top_k = self.cfg.retrieval.top_k
        # Re-Ranking holt in der ersten Stufe mehr Kandidaten und kürzt danach.
        rerank = self.cfg.retrieval.rerank
        first_stage_k = self.cfg.retrieval.rerank_candidates if rerank else top_k

        # Prio 3: Frage ggf. in Teilfragen zerlegen; jede suchen, Kandidaten
        # per RRF verschmelzen.
        queries = (await self._decompose(query, session_id)
                   if self.cfg.retrieval.query_decompose else [query])
        id_lists: list[list[str]] = []
        vec_scores: dict[str, float] = {}
        for q in queries:
            ids, scores = await self._candidate_ids(
                q, session, first_stage_k, session_id)
            id_lists.append(ids)
            vec_scores.update(scores)
        fused = (_rrf_merge(id_lists, first_stage_k) if len(id_lists) > 1
                 else (id_lists[0] if id_lists else []))
        hits = [(seg_id, vec_scores.get(seg_id, 0.0)) for seg_id in fused]
        if not hits:
            return None

        # Re-Ranker (Prio 1): Kandidaten per LLM neu sortieren, dann auf top_k.
        if rerank:
            hits = await self._rerank(query, hits, session_id)
        hits = hits[:top_k]

        # Graph-Expansion (Iteration D): Nachbar-Segmente über gemeinsame
        # Entitäten mitziehen — verbindet Evidenz über Sessions hinweg.
        if self.cfg.retrieval.graph_expansion:
            hit_ids = [seg_id for seg_id, _ in hits]
            neighbors = self.store.expand_by_graph(
                hit_ids, session.memory_id,
                limit=self.cfg.retrieval.graph_expansion_limit)
            fresh = [(nid, 0.0) for nid, _ in neighbors if nid not in set(hit_ids)]
            if fresh:
                hits = hits + fresh
                if self.bus is not None:
                    self.bus.emit("graph_expansion", session=session_id,
                                  seeds=len(hit_ids), added=len(fresh))

        budget = self.cfg.retrieval.max_injection_tokens
        used = 0
        parts = []
        for seg_id, score in hits:
            segment = self.store.get_segment(seg_id)
            if segment is None:
                continue
            text = segment.text
            tokens = self.counter.count_text(text)
            if used + tokens > budget:
                remaining = budget - used
                if remaining < _MIN_USEFUL_TOKENS:
                    break
                text = text[: remaining * 4]  # Heuristik ~4 Zeichen/Token
                tokens = remaining
            parts.append((segment, text, score))
            used += tokens
        if not parts:
            return None
        log.info("Swap-In für Session %s: %d Segment(e), ~%d Token",
                 session_id, len(parts), used)
        return build_retrieval_prompt(parts)

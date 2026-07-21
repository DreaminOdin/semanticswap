"""Extraction- & Summary-Engine: Sub-Agenten-Worker (PAD §1.3, Modus B).

Pro Segment laufen Summary- und Entity-Worker parallel (Map-Schritt);
die Nebenläufigkeit wird global über eine Semaphore (concurrency_limit) begrenzt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from ..config import SubAgentsConfig
from ..llm import LLMClient
from ..prompts import (ENTITY_EXTRACTION_PROMPT, PROFILE_PROMPT,
                       SUMMARIZATION_PROMPT)

log = logging.getLogger(__name__)

_MAX_TRIPLES_PER_SEGMENT = 15
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Entfernt Chain-of-Thought von Reasoning-Modellen (z. B. Qwen/DeepSeek):
    <think>...</think>-Blöcke sowie alles vor einem schließenden </think>."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return _THINK_BLOCK_RE.sub("", text).strip()


_PRIORITY_RE = re.compile(r"^\s*PRIORITY:\s*(high|low)\b[.:,]?\s*", re.IGNORECASE)


def split_priority(text: str) -> tuple[str, str]:
    """Trennt die PRIORITY-Zeile (ADR-011) vom Summary; Default konservativ high."""
    match = _PRIORITY_RE.match(text)
    if match:
        return match.group(1).lower(), text[match.end():].strip()
    return "high", text.strip()


@dataclass
class SegmentResult:
    summary: str = ""
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    priority: str = "high"


def parse_triples(raw: str) -> list[tuple[str, str, str]]:
    """Robustes Parsen der Worker-Antwort: JSON-Array, notfalls per Regex extrahiert."""
    text = raw.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        log.warning("Entity-Worker lieferte kein parsebares JSON: %.120s", raw)
        return []
    triples = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                s = str(item.get("subject", "")).strip()
                p = str(item.get("predicate", "")).strip()
                o = str(item.get("object", "")).strip()
                if s and p and o:
                    triples.append((s, p, o))
    # Harte Obergrenze: Modelle ignorieren das Prompt-Limit gelegentlich
    return triples[:_MAX_TRIPLES_PER_SEGMENT]


class ExtractionWorkers:
    def __init__(self, llm: LLMClient, cfg: SubAgentsConfig):
        self.llm = llm
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(max(1, cfg.concurrency_limit))

    async def _call(self, model: str, prompt: str) -> str:
        extra = {}
        if self.cfg.api_base:
            extra["api_base"] = self.cfg.api_base
        if self.cfg.api_key:
            extra["api_key"] = self.cfg.api_key
        async with self._semaphore:
            resp = await self.llm.complete(
                model, [{"role": "user", "content": prompt}], temperature=0.1, **extra
            )
        return strip_reasoning(resp["choices"][0]["message"]["content"] or "")

    async def summarize(self, segment_text: str) -> str:
        try:
            result = await self._call(
                self.cfg.tasks.summarization,
                SUMMARIZATION_PROMPT.format(segment=segment_text),
            )
            return result.strip()
        except Exception:
            log.exception("Summary-Worker fehlgeschlagen")
            return ""

    async def extract_entities(self, segment_text: str) -> list[tuple[str, str, str]]:
        try:
            raw = await self._call(
                self.cfg.tasks.entity_extraction,
                ENTITY_EXTRACTION_PROMPT.format(segment=segment_text),
            )
            return parse_triples(raw)
        except Exception:
            log.exception("Entity-Worker fehlgeschlagen")
            return []

    async def build_profile(self, summaries: list[str]) -> str:
        """Iteration B: destilliert aus den Segment-Summaries ein stehendes
        Nutzerprofil. Best effort — Fehler ergeben ein leeres Profil."""
        if not summaries:
            return ""
        try:
            joined = "\n".join(f"- {s}" for s in summaries if s.strip())
            return await self._call(
                self.cfg.tasks.summarization,
                PROFILE_PROMPT.format(summaries=joined))
        except Exception:
            log.exception("Profil-Worker fehlgeschlagen")
            return ""

    async def process_segment(self, segment_text: str) -> SegmentResult:
        raw_summary, triples = await asyncio.gather(
            self.summarize(segment_text),
            self.extract_entities(segment_text),
        )
        priority, summary = split_priority(raw_summary)
        return SegmentResult(summary=summary, triples=triples, priority=priority)

"""Gemeinsame Test-Fixtures. LLM-Aufrufe laufen gegen FakeLLM (ADR-007):
deterministisch, ohne Netzwerk. Verhalten wird über den Modellnamen gesteuert."""
from __future__ import annotations

import hashlib
import json

import pytest

from semanticswap.config import AppConfig


class FakeLLM:
    def __init__(self):
        self.calls: list[dict] = []

    def _content_for(self, model: str, messages: list[dict]) -> str:
        last = messages[-1].get("content", "") if messages else ""
        if isinstance(last, list):
            last = json.dumps(last)
        if "entity" in model:
            return json.dumps([
                {"subject": "User", "predicate": "arbeitet_an", "object": "Projekt X"},
                {"subject": "Projekt X", "predicate": "nutzt", "object": "Postgres"},
            ])
        if "summary" in model:
            return f"Kompakte Zusammenfassung ({len(last)} Zeichen Input)."
        if "synth" in model:
            return "Synthetisierte Konsolidierung."
        return f"echo:{last[-60:]}"

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        content = self._content_for(model, messages)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def complete_stream(self, model: str, messages: list[dict], **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs,
                           "stream": True})
        content = self._content_for(model, messages)
        half = max(1, len(content) // 2)
        for piece in (content[:half], content[half:]):
            yield {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece},
                             "finish_reason": None}],
            }
        yield {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    async def embed(self, model: str, texts: list[str], **kwargs) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([b / 255.0 for b in digest[:8]])
        return vectors


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def test_config() -> AppConfig:
    return AppConfig.model_validate({
        "gateway": {"port": 8080},
        "main_llm": {
            "provider": "fake",
            "model": "fake/main",
            "max_context_tokens": 120,
            "trigger_thresholds": [0.5],
        },
        "sub_agents": {
            "provider": "fake",
            "concurrency_limit": 2,
            "processing_mode": "batch",
            "tasks": {
                "summarization": "fake/summary",
                "entity_extraction": "fake/entity",
                "synthesizer": "fake/synth",
            },
        },
        "embedding": {"model": "fake/embed", "enabled": True},
        "storage": {"db_path": ":memory:", "vector_store": "sqlite_naive",
                    "chunk_size": 80},
        "compression": {"keep_recent_messages": 2},
        "retrieval": {"enabled": True, "top_k": 2, "min_score": 0.0,
                      "max_injection_tokens": 400},
    })

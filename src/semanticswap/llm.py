"""LLM-Zugriff über LiteLLM (ADR-005). Injizierbar, damit Tests ein Fake nutzen.

Alle Methoden liefern/erwarten OpenAI-kompatible Dict-Strukturen.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol


class LLMClient(Protocol):
    async def complete(self, model: str, messages: list[dict], **kwargs: Any) -> dict: ...

    def complete_stream(self, model: str, messages: list[dict],
                        **kwargs: Any) -> AsyncIterator[dict]: ...

    async def embed(self, model: str, texts: list[str],
                    **kwargs: Any) -> list[list[float]]: ...


class LiteLLMClient:
    """Produktiv-Implementierung; importiert litellm lazy (langsamer Import)."""

    async def complete(self, model: str, messages: list[dict], **kwargs: Any) -> dict:
        import litellm

        resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
        return resp.model_dump()

    async def complete_stream(self, model: str, messages: list[dict],
                              **kwargs: Any) -> AsyncIterator[dict]:
        import litellm

        stream = await litellm.acompletion(model=model, messages=messages,
                                           stream=True, **kwargs)
        async for chunk in stream:
            yield chunk.model_dump()

    async def embed(self, model: str, texts: list[str],
                    **kwargs: Any) -> list[list[float]]:
        import litellm

        resp = await litellm.aembedding(model=model, input=texts, **kwargs)
        data = resp["data"] if isinstance(resp, dict) else resp.data
        return [d["embedding"] if isinstance(d, dict) else d.embedding for d in data]

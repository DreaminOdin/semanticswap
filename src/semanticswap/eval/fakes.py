"""Deterministisches Offline-LLM für den Mechanik-Eval (kein Netzwerk, ADR-007).

Das Haupt-Modell arbeitet strikt EXTRAKTIV: Es beantwortet Referenzcode-Fragen
nur, wenn der Fakt im sichtbaren Kontext steht (Original-Message, ARCHIVE-Graph
oder Swap-In-Snippet). Der Offline-Recall misst damit exakt, ob das
Speichersystem den Fakt geliefert hat - nicht die Intelligenz des Modells.

Embeddings sind Bag-of-Words-Hash-Vektoren: Texte mit gemeinsamen Wörtern haben
höhere Cosine-Ähnlichkeit, sodass auch die Vektorsuche offline echtes
semantisches Verhalten zeigt.
"""
from __future__ import annotations

import hashlib
import json
import re

from ..tokens import plain_text

_DIM = 64
_FACT_RE = re.compile(r"Referenzcode für (.+?) lautet ([A-Z]+-\d+)")
_QUESTION_RE = re.compile(r"Referenzcode für (.+?)\?")


def bow_embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    for word in re.findall(r"[\wäöüß-]+", text.lower()):
        idx = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % _DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


def _segment_text(prompt: str) -> str:
    _, _, seg = prompt.partition("SEGMENT:")
    return seg.strip() or prompt


class OfflineLLM:
    def __init__(self):
        self.calls: list[dict] = []

    def _content_for(self, model: str, messages: list[dict]) -> str:
        last = plain_text(messages[-1].get("content")) if messages else ""
        if "summary" in model:
            words = _segment_text(last).split()[:18]
            return "PRIORITY: high\nZusammenfassung: " + " ".join(words)
        if "entity" in model:
            triples = [
                {"subject": topic, "predicate": "hat_referenzcode", "object": code}
                for topic, code in _FACT_RE.findall(_segment_text(last))
            ]
            return json.dumps(triples, ensure_ascii=False)
        if "synth" in model:
            return "Konsolidiert."

        # Haupt-Modell (extraktiv)
        question = _QUESTION_RE.search(last)
        if not question:
            return "Verstanden, notiert."
        topic = question.group(1).strip()
        context = "\n".join(plain_text(m.get("content")) for m in messages)
        patterns = [
            rf"Referenzcode für {re.escape(topic)} lautet ([A-Z]+-\d+)",
            rf"\({re.escape(topic)}\) -> hat_referenzcode -> \(([A-Z]+-\d+)\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return f"Der Referenzcode für {topic} lautet {match.group(1)}."
        return f"Dazu liegt mir kein Referenzcode für {topic} vor."

    def _response(self, model: str, content: str) -> dict:
        return {
            "id": "chatcmpl-offline",
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

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        self.calls.append({"model": model, "messages": messages})
        return self._response(model, self._content_for(model, messages))

    async def complete_stream(self, model: str, messages: list[dict], **kwargs):
        content = self._content_for(model, messages)
        yield {
            "id": "chatcmpl-offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content},
                         "finish_reason": None}],
        }
        yield {
            "id": "chatcmpl-offline",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    async def embed(self, model: str, texts: list[str], **kwargs) -> list[list[float]]:
        return [bow_embed(t) for t in texts]

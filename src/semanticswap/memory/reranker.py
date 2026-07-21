"""Cross-Encoder-Re-Ranker über ONNX (fastembed, Apache-2.0, kein PyTorch).

Bepunktet (Query, Passage)-Paare direkt — verlässlicher als ein kleines
Chat-Modell zum Listwise-Ranking zu überreden (das lieferte nur Teil-Listen).
Modell wird lazy geladen und pro Prozess gecacht; das Scoring ist synchron
(onnxruntime) und wird vom Aufrufer in einen Thread ausgelagert.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Englisch-optimiert, ~90 MB. Für mehrsprachigen Produktivbetrieb (DE) eignet
# sich z. B. 'jinaai/jina-reranker-v2-base-multilingual' (größer).
DEFAULT_CROSS_ENCODER = "Xenova/ms-marco-MiniLM-L-6-v2"

_cache: dict[str, "CrossEncoderReranker"] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log.info("Lade Cross-Encoder %s", self.model_name)
            self._model = TextCrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Relevanz-Score je Dokument (höher = relevanter). Synchron."""
        if not docs:
            return []
        model = self._ensure()
        return list(model.rerank(query, docs))


def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    name = model_name or DEFAULT_CROSS_ENCODER
    if name not in _cache:
        _cache[name] = CrossEncoderReranker(name)
    return _cache[name]

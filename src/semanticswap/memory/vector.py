"""Vektor-Store hinter Interface (ADR-004).

MVP-Implementierung: Embeddings als BLOB in SQLite, Cosine-Similarity via numpy.
LanceDB/ChromaDB können später dasselbe Interface implementieren.
"""
from __future__ import annotations

import numpy as np

from .store import Store


def to_blob(vector: list[float]) -> tuple[bytes, int]:
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


class NaiveVectorStore:
    def __init__(self, store: Store):
        self.store = store

    def add(self, segment_id: str, vector: list[float]) -> None:
        blob, dim = to_blob(vector)
        self.store.set_embedding(segment_id, blob, dim)

    def search(self, query: list[float], top_k: int = 5,
               session_id: str | None = None) -> list[tuple[str, float]]:
        """Exakte Cosine-Suche, numpy-vektorisiert (Prio 5): eine Matrix-
        Multiplikation statt Python-Schleife pro Vektor (~100x schneller,
        gleiches Ergebnis). Für sehr große Speicher wäre ein echter ANN-Index
        (HNSW/sqlite-vec) der nächste Schritt — dasselbe Interface (ADR-004),
        aber neue Abhängigkeit → PO-Entscheidung."""
        rows = self.store.all_embeddings(session_id)
        if not rows:
            return []
        q = np.asarray(query, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        ids = [r[0] for r in rows]
        # Alle Embeddings in eine (n, dim)-Matrix stapeln
        matrix = np.vstack([from_blob(blob, dim) for _, blob, dim in rows])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0  # Division durch 0 vermeiden (Score bleibt ~0)
        scores = (matrix @ q) / (norms * qn)
        # Top-k per argpartition (sublinear zur Sortierung der vollen Liste)
        k = min(top_k, len(ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(ids[i], float(scores[i])) for i in top_idx]

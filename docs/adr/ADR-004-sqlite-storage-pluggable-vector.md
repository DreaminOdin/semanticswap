# ADR-004: SQLite als Single-File-Speicher, Vektor-Store hinter Interface

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Das PAD empfiehlt SQLite (Graph) plus ChromaDB/LanceDB (Vektoren) — embedded,
ohne Server-Infrastruktur. Zusätzliche native Abhängigkeiten erhöhen aber das
Installationsrisiko (Windows/Linux/macOS) im MVP.

## Entscheidung
- **Eine SQLite-Datei** hält alles: Sessions, Hash-Kette, Original-Messages,
  Segmente (Volltext mit `source_segment_id`-Semantik), Graph-Tripel,
  Zusammenfassungen und Embeddings.
- Vektorsuche läuft im MVP über einen **naiven Store** (Embeddings als BLOB,
  Cosine-Similarity via numpy) hinter dem Interface `VectorStore`.
- LanceDB/ChromaDB werden als alternative `VectorStore`-Implementierungen
  ergänzt, sobald Datenmengen es erfordern (Konfigschlüssel
  `storage.vector_store`).

## Konsequenzen
- Null zusätzliche Infrastruktur, ein File = ein kompletter Speicherzustand
  (triviales Backup).
- Naive Cosine-Suche skaliert bis ~100k Segmente problemlos — weit über
  MVP-Bedarf; der Wechsel ist eine reine Implementierung des Interfaces.

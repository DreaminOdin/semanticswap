# ADR-002: MVP implementiert nur Modus B (Post-Response Batch)

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Das PAD definiert zwei Verarbeitungsmodi: A (On-the-fly Streaming-Ingestion mit
Tee-Filter) und B (Post-Response Map-Reduce-Batch). Modus A hat laut PAD hohe
Merge-Komplexität und niedrigere Schnitt-Präzision.

## Entscheidung
Der MVP implementiert ausschließlich **Modus B**: Nach Abschluss eines LLM-Turns
wird die Interaktion segmentiert, parallel von Workern (Entities, Summary)
verarbeitet und von einem Synthesizer konsolidiert. Die Pipeline-Schnittstelle
(`compression/pipeline.py`) ist so geschnitten, dass Modus A später als
alternative Ingestion-Quelle andocken kann.

## Konsequenzen
- Höchste Datenqualität (voller Interaktionskontext für Worker), geringere
  Komplexität im MVP.
- "Nacharbeitszeit" zwischen Turns existiert; akzeptabel, da Kompression
  asynchron läuft und erst beim übernächsten Request wirksam sein muss.

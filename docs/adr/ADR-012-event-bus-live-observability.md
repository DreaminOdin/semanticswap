# ADR-012: In-Process-Event-Bus und Live-Flowchart (Observability)

**Status:** Akzeptiert · **Datum:** 2026-07-16

## Kontext
Die Hintergrund-Verarbeitung (Kompression, Sub-Agenten, Swap-In) ist für den
User unsichtbar. Anforderung des PO: in Echtzeit sehen können, ob und welcher
Agent gerade im Hintergrund arbeitet — als Flowchart der Architektur.

## Entscheidung
1. Ein leichtgewichtiger **In-Process-Event-Bus** (`events.py`): Kernpfade
   (Gateway, Retrieval, Tool-Zyklus, Monitor, Kompressions-Pipeline,
   Sub-Agenten) emittieren typisierte Events; Subscriber erhalten sie über
   `asyncio.Queue`s, die letzten 200 Events werden für Late-Joiner gepuffert.
   Emission ist fire-and-forget und darf den Inferenz-Pfad nie blockieren
   (volle Subscriber-Queues verlieren Events, kein Backpressure).
2. Transport zum Browser über **Server-Sent-Events** (`GET /ui/events`,
   Replay des Puffers + Live-Stream, Heartbeat alle 15 s).
3. Die GUI-Seite **`/ui/flow`** rendert die PAD-Architektur als SVG-Flowchart;
   Knoten pulsieren bei Aktivität, ein Zähler zeigt aktive Sub-Agenten, ein
   Log listet die Events. Reines Anzeige-Feature: Kern bleibt headless
   (ADR-006), ohne GUI keinerlei Verhaltensänderung.

## Konsequenzen
- Volle Transparenz über die asynchrone Verarbeitung, auch als Debugging-Hilfe.
- Events sind in-memory und flüchtig (kein Audit-Log); Persistenz wäre ein
  eigenes ADR.
- Overhead pro Event: ein Dict + Queue-Put — vernachlässigbar.

## Nachtrag 2026-07-19: Hybrid-Retrieval-Knoten

Mit der Hybrid-Suche (M8 Iteration A: FTS5-Volltext + Vektoren via RRF)
wurde das Flowchart erweitert:

1. **Neuer Knoten `node-retrieval`** („Hybrid-Retrieval · Vektor + Volltext ·
   RRF", links unten): Die Speicher-Suche ist jetzt als eigene Komponente
   sichtbar statt im Memory-Knoten unterzugehen. Kanten: Semantic Memory →
   Hybrid-Retrieval (liest den Speicher) und gestrichelt zurück zum Gateway
   (Snippet-Injection in den Upstream).
2. **Neues Event `retrieval_search`** (emittiert vom `Retriever`, der dafür
   optional den Event-Bus erhält): Felder `mode` („hybrid"/„vector"),
   `vec`/`kw`/`fused` = Trefferzahlen der Teil-Suchen und der Fusion. Die
   Events `swap_in` und `tool_call` pulsieren jetzt ebenfalls den
   Retrieval-Knoten (vorher: Memory-Knoten).
3. Memory-Knoten-Untertitel aktualisiert („… · FTS"), SVG-`<title>`-Tooltips
   für beide Knoten.

Robustheits-Detail: Fällt das Embedding-Backend aus, degradiert die
Hybrid-Suche zur reinen Volltextsuche — im Event sichtbar als `vec=0`.

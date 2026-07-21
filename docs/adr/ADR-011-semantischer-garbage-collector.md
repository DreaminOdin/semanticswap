# ADR-011: Semantischer Garbage Collector über Prioritäten (Deep Archive)

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Nicht alle archivierten Inhalte sind langfristig relevant (PAD §5.2): Smalltalk
und transiente Details blähen den ARCHIVE-Prompt auf und verschlechtern die
Kompressions-Ratio (Eval 2026-07-15: 4.4:1 bei Ziel 5:1).

## Entscheidung
1. Der Summary-Worker vergibt beim Komprimieren pro Segment eine **Priorität**
   (`high` | `low`) über eine erste Antwortzeile `PRIORITY: ...`
   (nicht parsebar ⇒ konservativ `high`).
   - *high*: Fakten über den User, Projektziele, Entscheidungen, Referenzdaten.
   - *low*: Smalltalk, transiente Debugging-Schritte, Füllmaterial.
2. **Nichts wird gelöscht.** Low-Priority-Segmente, die nicht zu den letzten N
   Segmenten gehören (`compression.low_priority_visible`, Default 3), werden im
   ARCHIVE-Prompt zu einer einzelnen Sammelzeile zusammengefasst
   ("Deep Archive"). Volltext, Embeddings und Tripel bleiben erhalten und sind
   weiterhin über Swap-In (ADR-008) und das Retrieval-Tool (ADR-010) abrufbar.
3. Die Priorität wird pro Segment persistiert (Spalte `priority`,
   Mini-Migration wie ADR-009).

## Konsequenzen
- Der aktive ARCHIVE-Prompt wächst nur noch mit relevantem Inhalt; die Ratio
  verbessert sich mit der Verlaufslänge weiter.
- Fehleinschätzungen des Workers sind unkritisch: Deep-Archive-Inhalte sind
  nicht verloren, nur nicht mehr aktiv sichtbar.
- Echte Lösch-/TTL-Semantik (Speicherplatz) bleibt ein späteres ADR.

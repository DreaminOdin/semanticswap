# ADR-000: Architektur-Entscheidungen werden als ADRs dokumentiert

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Das Projekt SemanticSwap entsteht iterativ aus einem PAD. Architektur-Entscheidungen
(Sprache, Speicher, Verarbeitungsmodi) müssen nachvollziehbar und revidierbar sein.

## Entscheidung
Jede architekturrelevante Entscheidung wird als nummeriertes ADR in `docs/adr/`
abgelegt (Format: Kontext → Entscheidung → Konsequenzen → Alternativen).
ADRs werden nie gelöscht, sondern durch neue ADRs mit Status "Ersetzt durch ADR-xxx"
abgelöst.

## Konsequenzen
- Entscheidungen sind auditierbar; spätere Rewrites (z. B. Go statt Python) haben
  einen dokumentierten Ausgangspunkt.
- Kleiner Mehraufwand pro Entscheidung (~10 Minuten), bewusst in Kauf genommen.

# ADR-006: Headless-Core mit Admin-API, GUI als optionaler Layer

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Der Proxy soll als reiner Dienst (headless) betreibbar sein — Server, Docker,
CI — aber optional eine GUI für bessere UX bieten (Sessions/Speicher einsehen,
Kompressions-Statistiken, Graph-Visualisierung, Konfiguration).

## Entscheidung
- Der **Kern ist strikt headless**: Start via `uvicorn`/CLI, Konfiguration über
  `config.yaml` + Umgebungsvariablen, Beobachtbarkeit über strukturierte Logs.
- Alle Zustandsinformationen werden über eine **Admin-API** (`/admin/...`,
  read-only im MVP: Sessions, Statistiken, Segmente, Tripel) exponiert.
- Eine spätere **Web-GUI (Phase 4+)** ist ein eigenständiges Frontend, das
  ausschließlich die Admin-API konsumiert. Keine GUI-Abhängigkeit im Kern;
  der Proxy ist ohne GUI zu 100 % funktionsfähig.

## Konsequenzen
- Saubere Trennung: GUI kann als separates Paket/Container ausgeliefert oder
  weggelassen werden.
- Die Admin-API wird ab M2 mitentwickelt (zunächst minimal), damit die
  GUI-Grenze von Anfang an trägt.

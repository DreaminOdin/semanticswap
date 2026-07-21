# ADR-001: Python 3.11+ mit FastAPI und asyncio als MVP-Plattform

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Das PAD nennt Python (FastAPI) oder Go. Der Proxy muss voll asynchron sein
(TTFT-Schutz, Hintergrund-Worker), gleichzeitig zählt Iterationsgeschwindigkeit
im MVP mehr als minimale Proxy-Latenz.

## Entscheidung
MVP in **Python 3.11+ / FastAPI / asyncio**. Alle I/O-Pfade non-blocking;
Sub-Agenten-Jobs laufen über eine `asyncio.Queue` mit Worker-Pool.

## Konsequenzen
- Zugriff auf LiteLLM (ADR-005) und das gesamte Python-LLM-Ökosystem.
- Proxy-Overhead im einstelligen Millisekundenbereich ist akzeptiert; ein
  Go-Rewrite des Gateways bleibt Option, falls Messungen (M5) es erfordern.

## Alternativen
- **Go:** geringere Latenz, aber kein LiteLLM-Äquivalent, langsamere Iteration.

# ADR-005: LiteLLM als einzige Provider-Abstraktion

**Status:** Akzeptiert · **Datum:** 2026-07-15

## Kontext
Haupt-LLM (Cloud oder lokal) und Sub-Agenten (typisch Ollama) müssen über
verschiedene Provider ansprechbar sein, ohne eigene API-Anbindungen zu schreiben.

## Entscheidung
Alle LLM- und Embedding-Aufrufe laufen über **LiteLLM** (`acompletion`,
`aembedding`). Modellnamen in der Config verwenden LiteLLM-Notation
(z. B. `gpt-4o`, `ollama/llama3:8b-instruct`).

## Konsequenzen
- Provider-Wechsel ist reine Konfiguration.
- In Tests wird LiteLLM durch ein Fake gemockt — kein Netzwerkzugriff nötig.
- Abhängigkeit von einem großen Paket; akzeptiert, da es genau die
  PAD-Anforderung "Dutzende Provider auf OpenAI-Format" erfüllt.

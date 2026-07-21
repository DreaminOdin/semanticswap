# Quickstart: SemanticSwap as a drop-in memory proxy (Docker)

SemanticSwap sits between your OpenAI-compatible client and your local LLM
(Ollama). It transparently compresses long conversations into a semantic
memory (SQLite + knowledge graph + vectors) and swaps relevant originals back
in — so your model keeps "remembering" far beyond its context window.
Everything runs locally; no cloud keys.

## 1. Prerequisites

- Docker
- [Ollama](https://ollama.com) running on the host (default port 11434) with:
  ```
  ollama pull qwen2.5:7b        # default chat + worker model
  ollama pull nomic-embed-text  # embeddings for semantic retrieval
  ```

## 2. Run

```
docker build -t semanticswap .
docker run -d --name semanticswap -p 127.0.0.1:8080:8080 \
  -v semanticswap-data:/data \
  --add-host host.docker.internal:host-gateway \
  semanticswap
```

Check: `curl http://127.0.0.1:8080/health` → `{"status":"ok"}`.
Live GUI with chat + observability flowchart: http://127.0.0.1:8080/ui/studio

## 3. Point your client at it (the only change you need)

Wherever your tool asks for an OpenAI-compatible **base URL**, use
`http://127.0.0.1:8080/v1` instead of your Ollama/OpenAI endpoint.

| Client | Setting |
|---|---|
| **Open WebUI** | Admin → Connections → OpenAI API: `http://127.0.0.1:8080/v1`, any non-empty API key |
| **SillyTavern** | API: Chat Completion (OpenAI) → Custom Endpoint: `http://127.0.0.1:8080/v1` |
| **Python (openai SDK)** | `OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")` |
| **curl** | `curl http://127.0.0.1:8080/v1/chat/completions -d '{"model":"qwen2.5:7b","messages":[...]}'` |

Model selection is passed through: any model name your Ollama serves works
(`"model": "llama3.1:8b"` etc.). Streaming (`"stream": true`) is supported and
keeps flowing through edge proxies (built-in keepalives).

## 4. Custom configuration

Mount your own config (see `config.docker.yaml` for all options — models,
compression thresholds, retrieval, auth):

```
docker run ... -v ./my-config.yaml:/app/config.yaml semanticswap
```

Enable auth (recommended before exposing the port beyond localhost):

```
docker run ... -e SEMANTICSWAP_AUTH_USER=me -e SEMANTICSWAP_AUTH_PASSWORD=secret semanticswap
```

Clients then send the password as their OpenAI API key.

## 5. What you get

- **Context savings**: archived history is replaced by a compact ARCHIVE
  prompt; originals stay retrievable (swap-in + an agentic retrieval tool).
- **Observability**: `/ui/studio` shows KPIs and a live flowchart of every
  request, compression job, and memory retrieval as it happens.
- **One file**: the whole memory lives in a single SQLite database under
  `/data` — back it up, move it, inspect it.

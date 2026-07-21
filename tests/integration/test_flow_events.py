"""ADR-012: Live-Observability - Events über den vollen Zyklus + SSE + Flow-Seite."""
import json

import httpx
import pytest

from semanticswap.gateway import create_app

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
LONG = "Datenbankmigration nach Postgres, Connection-Pooling ist der Blocker. " * 6


@pytest.fixture
def app(fake_llm, test_config):
    return create_app(test_config, llm=fake_llm)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        yield c


@pytest.mark.asyncio
async def test_full_cycle_emits_expected_events(client, app):
    m = [SYSTEM, {"role": "user", "content": "Thema A: " + LONG}]
    r1 = await client.post("/v1/chat/completions", json={"model": "m", "messages": m})
    a1 = r1.json()["choices"][0]["message"]["content"]
    m2 = m + [{"role": "assistant", "content": a1},
              {"role": "user", "content": "Thema B: " + LONG}]
    await client.post("/v1/chat/completions", json={"model": "m", "messages": m2})
    await app.state.compressor.drain()

    types = [e["type"] for e in app.state.events.history]
    for expected in ["request", "session", "main_llm_start", "main_llm_done",
                     "monitor", "compression_enqueued", "compression_start",
                     "subagent_start", "subagent_done", "archive_updated"]:
        assert expected in types, f"Event {expected} fehlt: {types}"
    # Reihenfolge: Kompression erst nach Antwort (asynchron, PAD Modus B)
    assert types.index("main_llm_done") < types.index("compression_start")


@pytest.mark.asyncio
async def test_sse_stream_replays_history(client, app):
    await client.post("/v1/chat/completions", json={
        "model": "m", "messages": [SYSTEM, {"role": "user", "content": "Hi"}]})

    # replay_only: endlicher Stream (httpx-ASGI puffert Antworten vollständig)
    resp = await client.get("/ui/events", params={"replay_only": "true"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    received = [json.loads(line[6:]) for line in resp.text.splitlines()
                if line.startswith("data: ")]
    assert len(received) >= 3
    assert received[0]["type"] == "request"
    assert {"session", "main_llm_start"} & {e["type"] for e in received}


@pytest.mark.asyncio
async def test_flow_page_renders(client):
    resp = await client.get("/ui/flow")
    assert resp.status_code == 200
    assert "EventSource('/ui/events')" in resp.text
    assert "Sub-Agenten" in resp.text
    assert "node-mainllm" in resp.text

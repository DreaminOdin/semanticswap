"""Integration (M1-M3): voller Zyklus Request -> Session -> Kompression -> Pruning.

Läuft komplett gegen FakeLLM über ASGI-Transport - kein Netzwerk, keine echten
Modelle (ADR-007).
"""
import httpx
import pytest

from semanticswap.gateway import create_app
from semanticswap.prompts import ARCHIVE_HEADER, RETRIEVAL_HEADER

LONG = "Datenbankmigration nach Postgres, Connection-Pooling ist der Blocker. " * 6


@pytest.fixture
def app(fake_llm, test_config):
    return create_app(test_config, llm=fake_llm)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        yield c


def chat_body(messages, **extra):
    return {"model": "irrelevant-client-model", "messages": messages, **extra}


SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


@pytest.mark.asyncio
async def test_m1_passthrough_and_session_header(client, fake_llm):
    resp = await client.post("/v1/chat/completions",
                             json=chat_body([SYSTEM, {"role": "user", "content": "Hallo"}]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"].startswith("echo:")
    assert "x-semanticswap-session" in resp.headers
    # Upstream ging an das konfigurierte Haupt-LLM, nicht an das Client-Modell
    assert fake_llm.calls[0]["model"] == "fake/main"


@pytest.mark.asyncio
async def test_streaming_keepalive_during_quiet_phase(fake_llm, test_config,
                                                      monkeypatch):
    # Regression 2026-07-16: Edge-Proxys (Vercel) kappen stille Verbindungen
    # nach ~60 s. Bei großen Prompts fließen während der Prompt-Verarbeitung
    # keine Chunks - der Stream muss Keepalive-Kommentare senden.
    import asyncio

    from semanticswap import gateway as gw

    monkeypatch.setattr(gw, "STREAM_KEEPALIVE_SECONDS", 0.05)

    class SlowStartLLM:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def complete_stream(self, model, messages, **kw):
            await asyncio.sleep(0.2)  # stille Anlaufphase
            async for chunk in self._inner.complete_stream(model, messages, **kw):
                yield chunk

    app = create_app(test_config, llm=SlowStartLLM(fake_llm))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        resp = await c.post(
            "/v1/chat/completions",
            json=chat_body([SYSTEM, {"role": "user", "content": "Hi"}],
                           stream=True))
    assert resp.status_code == 200
    assert ": keepalive" in resp.text
    assert "data: [DONE]" in resp.text


@pytest.mark.asyncio
async def test_m1_streaming_sse(client, app):
    resp = await client.post(
        "/v1/chat/completions",
        json=chat_body([SYSTEM, {"role": "user", "content": "Stream bitte"}],
                       stream=True),
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "data: " in body
    assert "data: [DONE]" in body
    # Der Turn wurde nach dem Stream persistiert (system + user + assistant)
    sid = resp.headers["x-semanticswap-session"]
    assert app.state.store.message_count(sid) == 3


@pytest.mark.asyncio
async def test_m2_session_recognized_across_stateless_requests(client, app):
    m = [SYSTEM, {"role": "user", "content": "Frage 1"}]
    r1 = await client.post("/v1/chat/completions", json=chat_body(m))
    sid = r1.headers["x-semanticswap-session"]
    answer = r1.json()["choices"][0]["message"]["content"]

    m2 = m + [{"role": "assistant", "content": answer},
              {"role": "user", "content": "Frage 2"}]
    r2 = await client.post("/v1/chat/completions", json=chat_body(m2))
    assert r2.headers["x-semanticswap-session"] == sid
    assert len(app.state.store.list_sessions()) == 1


async def run_two_turns_and_compress(client, app):
    """Gemeinsamer Ablauf für M3/M4: zwei lange Turns, dann Kompression abwarten."""
    m = [SYSTEM, {"role": "user", "content": "Thema A: " + LONG}]
    r1 = await client.post("/v1/chat/completions", json=chat_body(m))
    sid = r1.headers["x-semanticswap-session"]
    a1 = r1.json()["choices"][0]["message"]["content"]

    m2 = m + [{"role": "assistant", "content": a1},
              {"role": "user", "content": "Thema B: " + LONG}]
    r2 = await client.post("/v1/chat/completions", json=chat_body(m2))
    a2 = r2.json()["choices"][0]["message"]["content"]

    await app.state.compressor.drain()
    m3 = m2 + [{"role": "assistant", "content": a2},
               {"role": "user", "content": "Worum ging es bei Thema A?"}]
    return sid, m3


@pytest.mark.asyncio
async def test_m3_swap_out_and_virtual_pruning(fake_llm, test_config):
    # Retrieval deaktiviert, damit exakt das Pruning (ohne Injection) messbar ist
    cfg = test_config.model_copy(deep=True)
    cfg.retrieval.enabled = False
    app = create_app(cfg, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        sid, m3 = await run_two_turns_and_compress(client, app)

        session = app.state.store.get_session(sid)
        assert session.archived_upto > 0
        assert ARCHIVE_HEADER in session.archive_prompt
        assert app.state.store.get_segments(sid)

        # Turn 3: Client schickt weiter den VOLLEN Verlauf (stateless)
        r3 = await client.post("/v1/chat/completions", json=chat_body(m3))
        assert r3.status_code == 200
        assert r3.headers["x-semanticswap-session"] == sid

        upstream = [c for c in fake_llm.calls if c["model"] == "fake/main"][-1]["messages"]
        # Virtuelles Pruning: Upstream ist kürzer als der Client-Verlauf ...
        assert len(upstream) < len(m3)
        # ... Original-System-Prompt bleibt erhalten, ARCHIVE-Prompt ist injiziert
        assert upstream[0] == SYSTEM
        assert ARCHIVE_HEADER in upstream[1]["content"]
        # ... und die letzte User-Frage geht unverändert durch
        assert upstream[-1]["content"] == "Worum ging es bei Thema A?"


@pytest.mark.asyncio
async def test_m4_swap_in_injects_original_snippets(client, app, fake_llm):
    sid, m3 = await run_two_turns_and_compress(client, app)
    session = app.state.store.get_session(sid)
    assert session.archived_upto > 0

    r3 = await client.post("/v1/chat/completions", json=chat_body(m3))
    assert r3.status_code == 200

    upstream = [c for c in fake_llm.calls if c["model"] == "fake/main"][-1]["messages"]
    # Injection steht direkt nach System- und ARCHIVE-Prompt
    assert ARCHIVE_HEADER in upstream[1]["content"]
    assert RETRIEVAL_HEADER in upstream[2]["content"]
    # und enthält den ORIGINAL-Volltext des archivierten Segments
    assert "Thema A: " + LONG[:40] in upstream[2]["content"]

    # Die Injection ist flüchtig: sie wird nie in der Session-Historie persistiert
    stored = app.state.store.get_messages(sid)
    assert all(RETRIEVAL_HEADER not in m.content for m in stored)


@pytest.mark.asyncio
async def test_m4_no_injection_without_archive(client, app, fake_llm):
    m = [SYSTEM, {"role": "user", "content": "kurze frage"}]
    await client.post("/v1/chat/completions", json=chat_body(m))
    upstream = [c for c in fake_llm.calls if c["model"] == "fake/main"][-1]["messages"]
    assert all(RETRIEVAL_HEADER not in (msg.get("content") or "") for msg in upstream)


@pytest.mark.asyncio
async def test_admin_api_exposes_state(client, app):
    m = [SYSTEM, {"role": "user", "content": "Hallo Admin"}]
    r = await client.post("/v1/chat/completions", json=chat_body(m))
    sid = r.headers["x-semanticswap-session"]

    stats = (await client.get("/admin/stats")).json()
    assert stats["sessions"] == 1
    assert stats["messages"] == 3

    sessions = (await client.get("/admin/sessions")).json()
    assert sessions[0]["id"] == sid

    detail = (await client.get(f"/admin/sessions/{sid}")).json()
    assert detail["messages"] == 3
    assert detail["archived_upto"] == 0

    assert (await client.get("/admin/sessions/unbekannt")).status_code == 404


@pytest.mark.asyncio
async def test_explicit_session_header_wins(client, app):
    m = [SYSTEM, {"role": "user", "content": "Hallo"}]
    r = await client.post("/v1/chat/completions", json=chat_body(m),
                          headers={"x-session-id": "kunde-42"})
    assert r.headers["x-semanticswap-session"] == "kunde-42"
    assert app.state.store.get_session("kunde-42") is not None


@pytest.mark.asyncio
async def test_empty_messages_rejected(client):
    r = await client.post("/v1/chat/completions", json=chat_body([]))
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_client_model_choice(fake_llm, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.main_llm.allow_client_model = True
    cfg.main_llm.client_model_prefix = "openai/"
    app = create_app(cfg, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        msgs = [SYSTEM, {"role": "user", "content": "hi"}]
        # Client wählt ein Modell -> Prefix wird vorangestellt
        await c.post("/v1/chat/completions",
                     json={"model": "gemma4:26b", "messages": msgs})
        assert fake_llm.calls[-1]["model"] == "openai/gemma4:26b"
        # Platzhalter "semanticswap" -> konfigurierter Default
        await c.post("/v1/chat/completions",
                     json={"model": "semanticswap", "messages": msgs})
        assert fake_llm.calls[-1]["model"] == "fake/main"


@pytest.mark.asyncio
async def test_client_model_ignored_by_default(client, fake_llm):
    await client.post("/v1/chat/completions", json=chat_body(
        [SYSTEM, {"role": "user", "content": "hi"}]))
    # allow_client_model=False: Client-Modell wird ignoriert
    assert fake_llm.calls[-1]["model"] == "fake/main"


@pytest.mark.asyncio
async def test_root_redirects_to_gui_and_v1_explains_itself(client):
    r = await client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/studio"

    info = await client.get("/v1")
    assert info.status_code == 200
    assert "chat/completions" in info.json()["api"]["chat"]

"""M6.1: Active Memory Retrieval - LLM-gesteuerter Swap-In (ADR-010)."""
import json

import httpx
import pytest

from semanticswap.gateway import create_app
from semanticswap.memory.active import TOOL_NAME, ActiveRetrieval
from semanticswap.memory.retrieval import Retriever
from semanticswap.memory.store import Segment, Store
from semanticswap.memory.vector import NaiveVectorStore
from semanticswap.tokens import TokenCounter

from conftest import FakeLLM

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
LONG = "Datenbankmigration nach Postgres, Connection-Pooling ist der Blocker. " * 6


class ToolCallingLLM(FakeLLM):
    """Haupt-Modell, das das Archiv-Tool nutzt, sobald es angeboten wird."""

    def _base(self, model: str) -> dict:
        return {"id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
                "model": model, "choices": []}

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        if model == "fake/main" and kwargs.get("tools"):
            self.calls.append({"model": model, "messages": messages,
                               "kwargs": kwargs})
            tool_results = [m for m in messages if m.get("role") == "tool"]
            resp = self._base(model)
            if tool_results:
                resp["choices"] = [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant",
                                                "content": "Archivdetails gefunden: "
                                                + tool_results[-1]["content"][:200]}}]
            else:
                resp["choices"] = [{"index": 0, "finish_reason": "tool_calls",
                                    "message": {"role": "assistant", "content": None,
                                                "tool_calls": [{
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": TOOL_NAME,
                                                        "arguments": json.dumps(
                                                            {"query": "Thema A"}),
                                                    }}]}}]
            return resp
        return await super().complete(model, messages, **kwargs)


async def run_two_turns(client, app):
    m = [SYSTEM, {"role": "user", "content": "Thema A: " + LONG}]
    r1 = await client.post("/v1/chat/completions", json={"model": "m", "messages": m})
    a1 = r1.json()["choices"][0]["message"]["content"]
    m2 = m + [{"role": "assistant", "content": a1},
              {"role": "user", "content": "Thema B: " + LONG}]
    r2 = await client.post("/v1/chat/completions", json={"model": "m", "messages": m2})
    a2 = r2.json()["choices"][0]["message"]["content"]
    await app.state.compressor.drain()
    sid = r1.headers["x-semanticswap-session"]
    return sid, m2 + [{"role": "assistant", "content": a2},
                      {"role": "user", "content": "Details zu Thema A bitte."}]


@pytest.mark.asyncio
async def test_tool_cycle_resolves_and_stays_invisible(test_config):
    llm = ToolCallingLLM()
    app = create_app(test_config, llm=llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as client:
        sid, m3 = await run_two_turns(client, app)
        assert app.state.store.get_session(sid).archived_upto > 0

        r3 = await client.post("/v1/chat/completions",
                               json={"model": "m", "messages": m3})
        content = r3.json()["choices"][0]["message"]["content"]
        # Das Modell hat das Tool genutzt und Original-Details erhalten
        assert content.startswith("Archivdetails gefunden:")
        assert "RETRIEVED ARCHIVE SNIPPETS" in content

        # Der Tool-Zyklus lief proxy-intern: zweiter Main-Call sah die Tool-Message
        tool_round = [c for c in llm.calls
                      if c["model"] == "fake/main"
                      and any(m.get("role") == "tool" for m in c["messages"])]
        assert len(tool_round) == 1

        # ... aber in der Session-Historie ist nichts davon persistiert
        roles = {m.raw.get("role") for m in app.state.store.get_messages(sid)}
        assert roles <= {"system", "user", "assistant"}


class NeverAnsweringToolLLM(ToolCallingLLM):
    """Pathologischer Fall (LongMemEval-Smoke 2026-07-18): Das Modell ruft in
    JEDER Runde erneut das Tool auf und liefert nie Inhalt - erst ein Call
    ohne Tool-Angebot antwortet."""

    async def complete(self, model: str, messages: list[dict], **kwargs) -> dict:
        if model == "fake/main" and kwargs.get("tools"):
            self.calls.append({"model": model, "messages": messages,
                               "kwargs": kwargs})
            resp = self._base(model)
            resp["choices"] = [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"role": "assistant", "content": None,
                                            "tool_calls": [{
                                                "id": "call_x",
                                                "type": "function",
                                                "function": {
                                                    "name": TOOL_NAME,
                                                    "arguments": json.dumps(
                                                        {"query": "Thema A"}),
                                                }}]}}]
            return resp
        if model == "fake/main":
            resp = self._base(model)
            resp["choices"] = [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant",
                                            "content": "Finale Antwort ohne Tools."}}]
            return resp
        return await FakeLLM.complete(self, model, messages, **kwargs)


@pytest.mark.asyncio
async def test_exhausted_tool_rounds_force_final_answer(test_config):
    # Regression 2026-07-18: Verbrauchte das Modell alle Tool-Runden mit
    # weiteren Tool-Aufrufen, lieferte der Proxy eine LEERE Antwort aus.
    llm = NeverAnsweringToolLLM()
    app = create_app(test_config, llm=llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as client:
        sid, m3 = await run_two_turns(client, app)
        assert app.state.store.get_session(sid).archived_upto > 0

        r3 = await client.post("/v1/chat/completions",
                               json={"model": "m", "messages": m3})
        content = r3.json()["choices"][0]["message"]["content"]
        assert content == "Finale Antwort ohne Tools."


@pytest.mark.asyncio
async def test_no_tool_injection_when_client_has_tools(test_config):
    llm = ToolCallingLLM()
    app = create_app(test_config, llm=llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as client:
        sid, m3 = await run_two_turns(client, app)

        client_tool = {"type": "function",
                       "function": {"name": "my_harness_tool", "parameters": {}}}
        r3 = await client.post("/v1/chat/completions", json={
            "model": "m", "messages": m3, "tools": [client_tool],
        })
        assert r3.status_code == 200
        last_main = [c for c in llm.calls if c["model"] == "fake/main"][-1]
        tool_names = [t["function"]["name"] for t in last_main["kwargs"]["tools"]]
        assert tool_names == ["my_harness_tool"]  # kein Proxy-Tool beigemischt


@pytest.mark.asyncio
async def test_resolve_by_segment_id_and_fallback(fake_llm, test_config):
    store = Store(":memory:")
    session = store.create_session(session_id="s1")
    store.add_segment(Segment(id="seg_s1_0001_0002", session_id=session.memory_id,
                              start_idx=1, end_idx=2,
                              text="USER: Originaltext über Postgres."))
    retriever = Retriever(store, NaiveVectorStore(store), fake_llm,
                          TokenCounter(use_tiktoken=False), test_config)
    active = ActiveRetrieval(store, retriever, test_config)

    call = {"id": "c1", "function": {"name": TOOL_NAME, "arguments":
            json.dumps({"segment_id": "#seg_s1_0001_0002"})}}
    result = await active.resolve("s1", call)
    assert "Originaltext über Postgres" in result

    call_unknown = {"id": "c2", "function": {"name": TOOL_NAME, "arguments":
                    json.dumps({"segment_id": "seg_gibtsnicht"})}}
    result = await active.resolve("s1", call_unknown)
    assert "No matching archived content" in result

    call_broken = {"id": "c3", "function": {"name": TOOL_NAME,
                                            "arguments": "kein json"}}
    result = await active.resolve("s1", call_broken)
    assert "No matching archived content" in result


def test_should_enable_conditions(fake_llm, test_config):
    store = Store(":memory:")
    retriever = Retriever(store, NaiveVectorStore(store), fake_llm,
                          TokenCounter(use_tiktoken=False), test_config)
    active = ActiveRetrieval(store, retriever, test_config)
    session = store.create_session(session_id="s1")
    store.set_archive("s1", 3, "ARCHIV")
    session = store.get_session("s1")

    assert active.should_enable(session, {}) is True
    assert active.should_enable(session, {"stream": True}) is False
    assert active.should_enable(session, {"tools": [{}]}) is False
    assert active.should_enable(session, {"functions": [{}]}) is False
    assert active.should_enable(None, {}) is False

    fresh = store.create_session(session_id="s2")  # ohne Archiv
    assert active.should_enable(fresh, {}) is False

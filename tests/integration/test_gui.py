"""Phase 4: GUI-Routen (/ui) - read-only Anzeige-Layer (ADR-006)."""
import httpx
import pytest

from semanticswap.gateway import create_app

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


@pytest.fixture
async def client(fake_llm, test_config):
    app = create_app(test_config, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        yield c


@pytest.mark.asyncio
async def test_dashboard_renders(client):
    resp = await client.get("/ui")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text
    assert "Sessions" in resp.text


@pytest.mark.asyncio
async def test_session_detail_renders(client):
    r = await client.post("/v1/chat/completions", json={
        "model": "m", "messages": [SYSTEM, {"role": "user", "content": "Hallo GUI"}],
    })
    sid = r.headers["x-semanticswap-session"]

    detail = await client.get(f"/ui/sessions/{sid}")
    assert detail.status_code == 200
    assert sid in detail.text
    assert "ARCHIVE-Prompt" in detail.text

    dashboard = await client.get("/ui")
    assert sid in dashboard.text


@pytest.mark.asyncio
async def test_studio_combines_chat_and_flow(client):
    resp = await client.get("/ui/studio")
    assert resp.status_code == 200
    # Chat und Live-Flow auf einer Seite, responsive Grid
    assert "/v1/chat/completions" in resp.text
    assert "EventSource('/ui/events')" in resp.text
    assert "@media (min-width: 1100px)" in resp.text
    assert 'class="stats"' in resp.text


@pytest.mark.asyncio
async def test_studio_shows_default_model_on_top(client):
    # Das konfigurierte Standard-Modell steht als erste Option im Dropdown
    # und ist als "(Standard)" gekennzeichnet.
    resp = await client.get("/ui/studio")
    assert "Modell: fake/main (Standard)" in resp.text


@pytest.mark.asyncio
async def test_studio_auswertung_cards_with_tooltips(client):
    # Auswertungs-KPIs (Archiv-Ratio, Kontext-Ersparnis, Deep Archive) plus
    # Hover-Tooltips auf allen Kacheln.
    resp = await client.get("/ui/studio")
    assert "Archiv-Ratio" in resp.text
    assert "Kontext-Ersparnis" in resp.text
    assert "Deep Archive" in resp.text
    assert 'data-tip="' in resp.text


@pytest.mark.asyncio
async def test_theme_toggle_dark_and_solarized(client):
    # Umschalter Dark <-> Solarized Light, Wahl wird in localStorage gemerkt.
    resp = await client.get("/ui/studio")
    assert 'id="themeToggle"' in resp.text
    assert "solarized" in resp.text
    assert "localStorage" in resp.text


@pytest.mark.asyncio
async def test_flow_shows_hybrid_retrieval_node(client):
    # ADR-012-Nachtrag: eigener Flowchart-Knoten für die Hybrid-Suche,
    # gespeist vom retrieval_search-Event.
    resp = await client.get("/ui/flow")
    assert 'id="node-retrieval"' in resp.text
    assert "retrieval_search" in resp.text
    assert "Vektor + Volltext" in resp.text


@pytest.mark.asyncio
async def test_chat_uses_streaming(client):
    # Regression 2026-07-16: Nicht-streamende Antworten > ~60 s werden vom
    # Edge-Proxy (Vercel) gekappt (HTTP-Fehler bei großen Modellen/langen
    # Antworten). Der GUI-Chat muss deshalb streamen.
    resp = await client.get("/ui/studio")
    assert "stream: true" in resp.text
    assert "getReader()" in resp.text


@pytest.mark.asyncio
async def test_flow_reconnects_after_fatal_sse_error(client):
    # Regression 2026-07-16: Nach einem 502 (eingefrorener Server hinter
    # Vercel) gibt EventSource endgültig auf - die Seite muss selbst neu
    # verbinden, sonst bleibt der Live-Flow bis zum manuellen Reload tot.
    resp = await client.get("/ui/flow")
    assert "setTimeout(connect" in resp.text
    assert "es.close()" in resp.text


@pytest.mark.asyncio
async def test_model_dropdown_options_readable(client):
    # Regression 2026-07-16: Optionen im Dark-Mode weiß auf weiß — Chrome
    # rendert das Popup hell, sobald das select ein eigenes background hat.
    # Canvas/CanvasText folgen dem aktiven Farbschema des Browsers.
    resp = await client.get("/ui/studio")
    assert "#model option { background: Canvas; color: CanvasText; }" in resp.text


@pytest.mark.asyncio
async def test_chat_page_renders(client):
    resp = await client.get("/ui/chat")
    assert resp.status_code == 200
    assert "/v1/chat/completions" in resp.text
    assert "x-session-id" in resp.text


@pytest.mark.asyncio
async def test_unknown_session_page(client):
    resp = await client.get("/ui/sessions/gibt-es-nicht")
    assert resp.status_code == 200
    assert "existiert nicht" in resp.text


@pytest.mark.asyncio
async def test_root_path_prefixes_all_links(fake_llm, test_config):
    """Betrieb unter apimanufaktur.de/semanticswap: alle URLs tragen das Präfix."""
    cfg = test_config.model_copy(deep=True)
    cfg.gateway.root_path = "/semanticswap"
    app = create_app(cfg, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        studio = (await c.get("/ui/studio")).text
        assert "EventSource('/semanticswap/ui/events')" in studio
        assert "fetch('/semanticswap/v1/chat/completions'" in studio
        assert 'href="/semanticswap/ui/chat"' in studio
        assert "__PREFIX__" not in studio

        r = await c.get("/", follow_redirects=False)
        assert r.headers["location"] == "/semanticswap/ui/studio"


@pytest.mark.asyncio
async def test_gui_escapes_html(client):
    await client.post("/v1/chat/completions", json={
        "model": "m",
        "messages": [SYSTEM, {"role": "user", "content": "<script>alert(1)</script>"}],
    }, headers={"x-session-id": "xss-test"})
    detail = await client.get("/ui/sessions/xss-test")
    assert "<script>alert(1)</script>" not in detail.text

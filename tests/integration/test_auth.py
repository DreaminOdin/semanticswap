"""Basic-Auth-Schutz für GUI und API (Voraussetzung für öffentliches Deployment)."""
import base64

import httpx
import pytest

from semanticswap.gateway import create_app

SYSTEM = {"role": "system", "content": "sys"}


@pytest.fixture
async def secured_client(fake_llm, test_config):
    cfg = test_config.model_copy(deep=True)
    cfg.gateway.auth_user = "ivo"
    cfg.gateway.auth_password = "geheim"
    app = create_app(cfg, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        yield c


def basic(user: str, pw: str) -> dict:
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.mark.asyncio
async def test_without_credentials_everything_is_locked(secured_client):
    for path in ["/ui", "/ui/studio", "/admin/stats", "/v1/models"]:
        resp = await secured_client.get(path)
        assert resp.status_code == 401, path
        assert resp.headers["www-authenticate"].startswith("Basic")
    post = await secured_client.post("/v1/chat/completions", json={
        "model": "m", "messages": [SYSTEM, {"role": "user", "content": "hi"}]})
    assert post.status_code == 401


@pytest.mark.asyncio
async def test_health_stays_open(secured_client):
    assert (await secured_client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_basic_auth_grants_access(secured_client):
    assert (await secured_client.get("/ui", headers=basic("ivo", "geheim"))).status_code == 200
    assert (await secured_client.get("/ui", headers=basic("ivo", "falsch"))).status_code == 401


@pytest.mark.asyncio
async def test_openai_clients_use_password_as_api_key(secured_client):
    resp = await secured_client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [SYSTEM, {"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer geheim"},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"].startswith("echo:")


@pytest.mark.asyncio
async def test_no_auth_config_means_open(fake_llm, test_config):
    app = create_app(test_config, llm=fake_llm)  # ohne auth_user/password
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        assert (await c.get("/ui")).status_code == 200


# --- Auth v2 (docs/plan-auth-v2.md): Login-Seite, Geräte-Cookie, Rollen ----


@pytest.mark.asyncio
async def test_login_page_is_reachable_without_auth(secured_client):
    resp = await secured_client.get("/ui/login")
    assert resp.status_code == 200
    assert 'name="password"' in resp.text


@pytest.mark.asyncio
async def test_login_issues_family_device_cookie(secured_client):
    resp = await secured_client.post("/ui/login", data={"password": "geheim"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/studio"
    cookie = resp.cookies.get("semanticswap_device")
    assert cookie

    # Familie: Chat + Live-Ansicht erlaubt ...
    for path in ["/ui/studio", "/ui/flow", "/ui/chat", "/v1/models"]:
        r = await secured_client.get(path,
                                     cookies={"semanticswap_device": cookie})
        assert r.status_code == 200, path
    # ... Verwaltung nicht (Rollen, P4)
    for path in ["/ui", "/admin/stats"]:
        r = await secured_client.get(path,
                                     cookies={"semanticswap_device": cookie})
        assert r.status_code == 403, path


@pytest.mark.asyncio
async def test_wrong_login_registers_brake(secured_client):
    for _ in range(6):
        resp = await secured_client.post("/ui/login",
                                         data={"password": "falsch"})
    assert resp.status_code in (401, 429)
    resp = await secured_client.post("/ui/login", data={"password": "falsch"})
    assert resp.status_code == 429  # Bremse greift
    assert "Sekunden" in resp.text


@pytest.mark.asyncio
async def test_gui_paths_redirect_browsers_to_login(secured_client):
    resp = await secured_client.get(
        "/ui/studio", headers={"accept": "text/html"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/ui/login"
    # API-Pfade bleiben beim klassischen 401 (Clients erwarten das)
    resp = await secured_client.get("/v1/models")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_basic_auth_keeps_admin_access(secured_client):
    # Admin (Basic) sieht weiterhin alles — auch die Verwaltung
    assert (await secured_client.get(
        "/admin/stats", headers=basic("ivo", "geheim"))).status_code == 200


@pytest.mark.asyncio
async def test_tampered_cookie_is_ignored(secured_client):
    r = await secured_client.get(
        "/ui/studio", cookies={"semanticswap_device": "abc.def"})
    assert r.status_code in (303, 401)


@pytest.mark.asyncio
async def test_external_attempts_are_logged_with_origin_ip(secured_client):
    # Auth v2: Zugriffe außerhalb des Tailnets landen im Logbuch — mit der
    # echten Herkunfts-IP aus x-forwarded-for (von Vercel gesetzt).
    await secured_client.get("/v1/models",
                             headers={"x-forwarded-for": "203.0.113.55"})
    await secured_client.post("/ui/login", data={"password": "falsch"},
                              headers={"x-forwarded-for": "45.9.148.99"})

    resp = await secured_client.get("/admin/access-log",
                                    headers=basic("ivo", "geheim"))
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    blocked = next(e for e in entries if e["path"].endswith("/v1/models"))
    assert blocked["ip"] == "203.0.113.55"
    assert blocked["outcome"] == "blocked_401" or blocked["outcome"] == "no_auth"
    assert blocked["via"] == "edge"
    assert any(e["ip"] == "45.9.148.99" and e["outcome"] == "login_fail"
               for e in entries)


@pytest.mark.asyncio
async def test_access_log_html_page_is_admin_only(secured_client):
    await secured_client.get("/v1/models",
                             headers={"x-forwarded-for": "203.0.113.77"})
    # Admin sieht die Tabelle
    resp = await secured_client.get("/ui/access", headers=basic("ivo", "geheim"))
    assert resp.status_code == 200
    assert "203.0.113.77" in resp.text and "Zugriffs-Log" in resp.text
    # Familie (Cookie) darf nicht
    login = await secured_client.post("/ui/login", data={"password": "geheim"})
    cookie = login.cookies.get("semanticswap_device")
    r = await secured_client.get("/ui/access",
                                 cookies={"semanticswap_device": cookie})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_forwarded_header_never_grants_tailnet_trust(fake_llm, test_config,
                                                           monkeypatch):
    # Sicherheit: ein gefälschter x-forwarded-for mit Tailnet-IP darf NIE
    # als Tailnet-Vertrauen durchgehen — Vertrauen hängt an der TCP-Quelle.
    from semanticswap import gateway as gw

    cfg = test_config.model_copy(deep=True)
    cfg.gateway.auth_user = "ivo"
    cfg.gateway.auth_password = "geheim"
    cfg.gateway.trust_tailnet = True
    cfg.gateway.tailnet_users = ["ivo@example.com"]
    monkeypatch.setattr(gw, "_tailscale_whois",
                        lambda ip: (_ for _ in ()).throw(AssertionError(
                            "whois darf für Header-IP nicht aufgerufen werden")))
    app = create_app(cfg, llm=fake_llm)
    transport = httpx.ASGITransport(app=app)  # Peer = testclient, kein Tailnet
    async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
        r = await c.get("/admin/stats",
                        headers={"x-forwarded-for": "100.100.1.2"})
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_tailnet_device_of_owner_needs_no_password(fake_llm, test_config,
                                                         monkeypatch):
    # P3: Anfragen von Ivos Tailnet-Geräten (whois-Identität) ohne Passwort;
    # fremde/geteilte Tailnet-Geräte bleiben draußen.
    from semanticswap import gateway as gw

    cfg = test_config.model_copy(deep=True)
    cfg.gateway.auth_user = "ivo"
    cfg.gateway.auth_password = "geheim"
    cfg.gateway.trust_tailnet = True
    cfg.gateway.tailnet_users = ["ivo@example.com"]

    async def fake_whois(ip):
        return "ivo@example.com" if ip == "100.100.1.2" else "jan@example.com"

    monkeypatch.setattr(gw, "_tailscale_whois", fake_whois)
    app = create_app(cfg, llm=fake_llm)

    own = httpx.ASGITransport(app=app, client=("100.100.1.2", 1234))
    async with httpx.AsyncClient(transport=own, base_url="http://p") as c:
        assert (await c.get("/admin/stats")).status_code == 200

    foreign = httpx.ASGITransport(app=app, client=("100.77.7.7", 1234))
    async with httpx.AsyncClient(transport=foreign, base_url="http://p") as c:
        assert (await c.get("/admin/stats")).status_code == 401

"""API Gateway & Protocol Translator (PAD §1.1) + Admin-API (ADR-006).

Spiegelt die OpenAI-API (/v1/chat/completions). Der Client merkt nicht, dass
eine Middleware zwischengeschaltet ist. Pruning ist virtuell (ADR-003): Die
Historie des Clients bleibt unangetastet; nur der Upstream-Request an das
Haupt-LLM wird um den ARCHIVE-Prompt herum verkleinert.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from .compression.pipeline import BackgroundCompressor, CompressionPipeline
from .compression.workers import ExtractionWorkers
from .config import AppConfig
from .events import EventBus
from .llm import LiteLLMClient, LLMClient
from .memory.active import MAX_TOOL_ROUNDS, ActiveRetrieval
from .memory.retrieval import Retriever
from .memory.store import Session, Store
from .monitor import ContextMonitor
from .session.tracker import SessionTracker
from .tokens import TokenCounter

log = logging.getLogger(__name__)

# Edge-Proxys (z. B. Vercel) kappen stille Verbindungen nach ~60 s. Während
# Modell-Laden/Prompt-Verarbeitung fließen keine Chunks - so lange sendet der
# Stream SSE-Kommentare als Lebenszeichen (Clients ignorieren ':'-Zeilen).
STREAM_KEEPALIVE_SECONDS = 10.0


def _is_tailnet_ip(ip: str) -> bool:
    """Tailscale-CGNAT-Bereich. Funnel-Verkehr erreicht den Dienst dagegen
    als 127.0.0.1 und zählt bewusst NICHT als Tailnet (Auth v2, P3)."""
    import ipaddress

    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


_whois_cache: dict[str, str | None] = {}


async def _tailscale_whois(ip: str) -> str | None:
    """Tailscale-Identität (LoginName) hinter einer Tailnet-IP; None wenn
    unbekannt oder tailscale nicht verfügbar. Gecacht pro Prozess."""
    if ip in _whois_cache:
        return _whois_cache[ip]
    try:
        proc = await asyncio.create_subprocess_exec(
            "tailscale", "whois", "--json", ip,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
        login = (json.loads(out).get("UserProfile") or {}).get("LoginName")
    except Exception:
        login = None
    _whois_cache[ip] = login
    return login

_PASSTHROUGH_EXCLUDE = {"model", "messages", "stream"}


def build_upstream_messages(messages: list[dict], session: Session) -> list[dict]:
    """Virtuelles Pruning: archivierten Prefix durch ARCHIVE-Prompt ersetzen."""
    if session.archived_upto <= 0 or not session.archive_prompt:
        return messages
    upstream: list[dict] = []
    if messages and messages[0].get("role") == "system":
        upstream.append(messages[0])
    upstream.append({"role": "system", "content": session.archive_prompt})
    upstream.extend(messages[session.archived_upto:])
    return upstream


def create_app(cfg: AppConfig, llm: LLMClient | None = None) -> FastAPI:
    llm = llm or LiteLLMClient()
    store = Store(cfg.storage.db_path)
    events = EventBus()
    tracker = SessionTracker(store)
    counter = TokenCounter(cfg.main_llm.model)
    monitor = ContextMonitor(cfg.main_llm.max_context_tokens,
                             cfg.main_llm.trigger_thresholds)
    workers = ExtractionWorkers(llm, cfg.sub_agents)
    pipeline = CompressionPipeline(store, workers, counter, cfg, llm=llm, bus=events)
    compressor = BackgroundCompressor(pipeline)
    retriever = Retriever(store, pipeline.vectors, llm, counter, cfg,
                          bus=events)
    active = ActiveRetrieval(store, retriever, cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await compressor.drain()
        await compressor.stop()
        store.close()

    app = FastAPI(title="SemanticSwap Proxy", lifespan=lifespan)

    auth_user = cfg.gateway.auth_user or os.environ.get("SEMANTICSWAP_AUTH_USER")
    auth_password = (cfg.gateway.auth_password
                     or os.environ.get("SEMANTICSWAP_AUTH_PASSWORD"))
    if auth_user and auth_password:
        from pathlib import Path

        from .auth import (AccessLog, DeviceCookieSigner, LoginBrake,
                           load_or_create_secret)

        expected_basic = "Basic " + base64.b64encode(
            f"{auth_user}:{auth_password}".encode()).decode()
        expected_bearer = f"Bearer {auth_password}"  # OpenAI-Clients: api_key=Passwort

        if cfg.storage.db_path == ":memory:":
            cookie_secret = secrets.token_bytes(32)
            access_log = AccessLog(None)
        else:
            data_dir = Path(cfg.storage.db_path).parent
            cookie_secret = load_or_create_secret(data_dir / "auth_secret.key")
            access_log = AccessLog(data_dir / "access-log.jsonl")
        signer = DeviceCookieSigner(cookie_secret)
        brake = LoginBrake()
        app.state.device_signer = signer
        app.state.access_log = access_log
        rp = cfg.gateway.root_path or ""

        # Rolle "family" (Geräte-Cookie via Login-Seite): Chat + Live-Ansicht.
        # Verwaltung (/ui-Sessions, /admin) bleibt Admin (Basic/Bearer/Tailnet).
        family_paths = tuple(rp + p for p in (
            "/ui/studio", "/ui/flow", "/ui/chat", "/ui/events",
            "/v1/chat/completions", "/v1/models"))
        open_paths = {rp + "/health", "/health", rp + "/ui/login"}

        login_html = f"""<!doctype html><html lang="de"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SemanticSwap - Anmelden</title><style>
:root {{ color-scheme: light dark; }}
body {{ font-family: system-ui, sans-serif; display: flex; min-height: 100vh;
       align-items: center; justify-content: center; margin: 0; }}
form {{ display: flex; flex-direction: column; gap: .8rem; min-width: 18rem;
        padding: 2rem; border: 1px solid color-mix(in srgb, currentColor 25%, transparent);
        border-radius: 12px; }}
input[type=password] {{ padding: .6rem; font: inherit; border-radius: 8px;
        border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }}
button {{ padding: .6rem; font: inherit; border: none; border-radius: 8px;
          background: #2f81f7; color: #fff; cursor: pointer; }}
label {{ font-size: .9rem; }}
</style></head><body>
<form method="post" action="{rp}/ui/login">
  <b>SemanticSwap</b>
  <input type="password" name="password" placeholder="Passwort" autofocus required>
  <label><input type="checkbox" name="remember" checked> Dieses Gerät merken
   ({cfg.gateway.device_cookie_days} Tage)</label>
  <button type="submit">Anmelden</button>
</form></body></html>"""

        def _peer_ip(request: Request) -> str:
            """TCP-Quelle — nur DIESE für Tailnet-Vertrauen (nicht fälschbar)."""
            return request.client.host if request.client else "?"

        def _origin_ip(request: Request, peer: str) -> str:
            """Echte Herkunfts-IP fürs Logbuch: hinter Vercel steht sie in
            x-forwarded-for. NUR fürs Anzeigen/Loggen, NIE für Auth."""
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip()
            return request.headers.get("x-real-ip") or peer

        def _log_access(request: Request, peer: str, outcome: str,
                        role: str | None) -> None:
            ip = _origin_ip(request, peer)
            via = ("tailnet" if _is_tailnet_ip(peer)
                   else "edge" if request.headers.get("x-forwarded-for")
                   else "funnel")
            access_log.record(ip=ip, peer=peer, via=via, method=request.method,
                              path=request.url.path, outcome=outcome, role=role)
            # Verdächtige/abgewiesene Zugriffe zusätzlich live auf den Bus
            if outcome not in ("admin_basic", "family"):
                events.emit("access_denied", ip=ip, via=via,
                            path=request.url.path, outcome=outcome)

        @app.get(rp + "/ui/login", include_in_schema=False)
        async def login_page() -> Response:
            return Response(content=login_html, media_type="text/html; charset=utf-8")

        @app.post(rp + "/ui/login", include_in_schema=False)
        async def login_submit(request: Request) -> Response:
            peer = _peer_ip(request)
            ip = _origin_ip(request, peer)
            allowed, wait = brake.check(ip)
            if not allowed:
                events.emit("login", ip=ip, ok=False, throttled=True)
                _log_access(request, peer, "login_throttled", None)
                return Response(
                    status_code=429,
                    content=f"Zu viele Fehlversuche. Bitte {int(wait) + 1} "
                            f"Sekunden warten.",
                    media_type="text/plain; charset=utf-8")
            # Bewusst ohne python-multipart: das Login-Formular ist reines
            # application/x-www-form-urlencoded, stdlib reicht.
            from urllib.parse import parse_qs

            raw = (await request.body()).decode("utf-8", "replace")
            form = {k: v[0] for k, v in parse_qs(raw).items()}
            password = str(form.get("password", ""))
            if secrets.compare_digest(password, auth_password):
                brake.register_success(ip)
                events.emit("login", ip=ip, ok=True)
                _log_access(request, peer, "login_ok", "family")
                token = signer.issue("family",
                                     days=cfg.gateway.device_cookie_days)
                resp = Response(status_code=303,
                                headers={"location": rp + "/ui/studio"})
                max_age = (cfg.gateway.device_cookie_days * 86400
                           if form.get("remember") else None)
                # secure=False: Tailnet-Direktzugriff läuft über http;
                # Domain/Funnel sind ohnehin TLS-terminiert.
                resp.set_cookie("semanticswap_device", token, max_age=max_age,
                                httponly=True, samesite="lax")
                return resp
            brake.register_failure(ip)
            events.emit("login", ip=ip, ok=False)
            _log_access(request, peer, "login_fail", None)
            return Response(status_code=401, content="Falsches Passwort.",
                            media_type="text/plain; charset=utf-8")

        @app.middleware("http")
        async def auth_v2(request: Request, call_next):
            path = request.url.path
            if path in open_paths:  # Liveness + Login bleiben offen
                return await call_next(request)
            peer = _peer_ip(request)

            # Entscheidung treffen: (Aktion, outcome, role)
            provided = request.headers.get("authorization", "")
            action = outcome = None
            role: str | None = None
            if (secrets.compare_digest(provided, expected_basic)
                    or secrets.compare_digest(provided, expected_bearer)):
                action, outcome, role = "pass", "admin_basic", "admin"
            else:
                cookie_role = signer.verify(
                    request.cookies.get("semanticswap_device", ""))
                if cookie_role == "admin":
                    action, outcome, role = "pass", "admin_cookie", "admin"
                elif cookie_role == "family":
                    if path.startswith(family_paths):
                        action, outcome, role = "pass", "family", "family"
                    else:
                        action, outcome, role = "deny403", "family_denied_admin", "family"
                elif request.cookies.get("semanticswap_device"):
                    action, outcome = "challenge", "tampered_cookie"
                elif cfg.gateway.trust_tailnet and _is_tailnet_ip(peer):
                    login_name = await _tailscale_whois(peer)
                    if login_name and login_name in cfg.gateway.tailnet_users:
                        action, outcome, role = "pass_tailnet", "tailnet", "admin"
                    else:
                        action, outcome = "challenge", "tailnet_untrusted"
                else:
                    action, outcome = "challenge", "no_auth"

            # Loggen: alles außerhalb des vertrauten Tailnets (Jan + Fremde)
            if action != "pass_tailnet":
                _log_access(request, peer, outcome, role)

            if action in ("pass", "pass_tailnet"):
                return await call_next(request)
            if action == "deny403":
                return Response(
                    status_code=403,
                    content="Dieser Bereich ist dem Admin vorbehalten.",
                    media_type="text/plain; charset=utf-8")
            # challenge: Browser zur Login-Seite, API-Clients klassisch 401
            accept = request.headers.get("accept", "")
            if path.startswith(rp + "/ui") and "text/html" in accept:
                return Response(status_code=303,
                                headers={"location": rp + "/ui/login"})
            return Response(
                status_code=401,
                content="Zugang nur mit Passwort.",
                headers={"WWW-Authenticate": 'Basic realm="SemanticSwap"'},
            )

        @app.get(rp + "/admin/access-log", include_in_schema=False)
        async def access_log_json(limit: int = 100):
            return JSONResponse({"entries": access_log.recent(limit)})

        @app.get(rp + "/ui/access", include_in_schema=False)
        async def access_log_page() -> Response:
            from html import escape as _esc

            from .gui import _page

            rows = []
            for e in access_log.recent(200):
                ok = e.get("outcome") in ("admin_basic", "admin_cookie",
                                          "family", "login_ok")
                cls = "ok" if ok else "warn"
                rows.append(
                    f'<tr class="{cls}"><td>{_esc(e.get("ts", ""))}</td>'
                    f'<td>{_esc(str(e.get("ip", "")))}</td>'
                    f'<td>{_esc(str(e.get("via", "")))}</td>'
                    f'<td>{_esc(e.get("method", ""))}</td>'
                    f'<td>{_esc(e.get("path", ""))}</td>'
                    f'<td>{_esc(e.get("outcome", ""))}</td></tr>')
            body = (
                "<style>"
                "table{border-collapse:collapse;width:100%;font-size:.85rem}"
                "th,td{text-align:left;padding:.3rem .5rem;border-bottom:1px solid"
                " color-mix(in srgb,currentColor 15%,transparent);"
                "font-family:ui-monospace,monospace}"
                "tr.warn td{color:#d9a04a}"
                ".muted{opacity:.65}</style>"
                '<p class="muted">Zugriffe außerhalb deines Tailnets (Familie + '
                "Fremde). Warnfarbe = abgewiesen/verdächtig. Neueste zuerst, "
                "letzte 200; Vollverlauf in <code>data/access-log.jsonl</code> "
                'auf dem Server.</p>'
                "<table><tr><th>Zeit (UTC)</th><th>Herkunfts-IP</th><th>Weg</th>"
                "<th>Methode</th><th>Pfad</th><th>Ergebnis</th></tr>"
                + "".join(rows) + "</table>")
            return _page("Zugriffs-Log", body, rp)
    app.state.cfg = cfg
    app.state.store = store
    app.state.events = events
    app.state.tracker = tracker
    app.state.compressor = compressor
    app.state.pipeline = pipeline
    app.state.llm = llm

    def finalize_turn(session_id: str, request_messages: list[dict],
                      assistant_content: str) -> None:
        """Nach jedem Turn: Verlauf persistieren, Token prüfen, ggf. Kompression
        anstoßen (Modus B, asynchron - blockiert die Antwort nicht)."""
        full = request_messages + [{"role": "assistant", "content": assistant_content}]
        tracker.record(session_id, full)
        session = store.get_session(session_id)
        if session is None:
            return
        upstream = build_upstream_messages(full, session)
        result = monitor.check(counter.count_messages(upstream))
        events.emit("monitor", session=session_id, tokens=result.tokens,
                    ratio=round(result.ratio, 3),
                    triggered=result.triggered_threshold)
        if result.triggered_threshold is not None:
            queued = compressor.enqueue(session_id)
            if queued:
                events.emit("compression_enqueued", session=session_id)
            log.info(
                "Session %s: %d Token (%.0f%%) >= Schwelle %.0f%% -> Kompression %s",
                session_id, result.tokens, result.ratio * 100,
                result.triggered_threshold * 100,
                "eingeplant" if queued else "läuft bereits",
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages: list[dict] = body.get("messages", [])
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages must not be empty",
                                   "type": "invalid_request_error"}},
            )
        explicit_id = request.headers.get("x-session-id") or body.get("user")
        events.emit("request", messages=len(messages), stream=bool(body.get("stream")))
        resolution = tracker.resolve(messages, explicit_id=explicit_id)
        events.emit("session", session=resolution.session_id,
                    matched=resolution.matched_count, new=resolution.is_new,
                    fork=resolution.forked_from)
        session = store.get_session(resolution.session_id)
        upstream = build_upstream_messages(messages, session)  # type: ignore[arg-type]

        # Swap-In v1 (ADR-008): Original-Snippets temporär nach dem ARCHIVE-Prompt
        # injizieren; wird nie persistiert.
        if session and session.archived_upto > 0 and session.archive_prompt:
            injection = await retriever.build_injection(resolution.session_id, messages)
            events.emit("swap_in", session=resolution.session_id,
                        hit=injection is not None)
            if injection is not None:
                insert_at = 2 if messages and messages[0].get("role") == "system" else 1
                upstream = upstream[:insert_at] + [injection] + upstream[insert_at:]
        kwargs = {k: v for k, v in body.items() if k not in _PASSTHROUGH_EXCLUDE}
        kwargs.pop("user", None)
        if cfg.main_llm.api_base:
            kwargs["api_base"] = cfg.main_llm.api_base
        if cfg.main_llm.api_key:
            kwargs["api_key"] = cfg.main_llm.api_key
        model = cfg.main_llm.model
        requested = (body.get("model") or "").strip()
        if (cfg.main_llm.allow_client_model and requested
                and requested not in ("semanticswap", cfg.main_llm.model)):
            model = cfg.main_llm.client_model_prefix + requested
        headers = {"x-semanticswap-session": resolution.session_id}

        if body.get("stream"):
            async def sse():
                collected: list[str] = []
                events.emit("main_llm_start", session=resolution.session_id,
                            upstream_tokens=counter.count_messages(upstream))
                ait = llm.complete_stream(model, upstream, **kwargs).__aiter__()
                pending = None
                while True:
                    if pending is None:
                        pending = asyncio.ensure_future(anext(ait))
                    try:
                        chunk = await asyncio.wait_for(
                            asyncio.shield(pending),
                            timeout=STREAM_KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    except StopAsyncIteration:
                        break
                    pending = None
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            collected.append(delta["content"])
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                events.emit("main_llm_done", session=resolution.session_id)
                finalize_turn(resolution.session_id, messages, "".join(collected))

            return StreamingResponse(sse(), media_type="text/event-stream",
                                     headers=headers)

        # Active Memory Retrieval (ADR-010): Tool nur ohne Kollisionsrisiko anbieten
        call_kwargs = dict(kwargs)
        use_tool = active.should_enable(session, body)
        if use_tool:
            call_kwargs["tools"] = [active.tool_definition()]

        events.emit("main_llm_start", session=resolution.session_id,
                    upstream_tokens=counter.count_messages(upstream))
        response = await llm.complete(model, upstream, **call_kwargs)
        events.emit("main_llm_done", session=resolution.session_id)
        rounds = 0
        while use_tool and rounds < MAX_TOOL_ROUNDS:
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            ours = active.own_tool_calls(message)
            if not tool_calls or len(ours) != len(tool_calls):
                break
            # Tool-Zyklus läuft proxy-intern; der Client sieht ihn nie
            upstream = upstream + [message]
            for call in ours:
                result = await active.resolve(resolution.session_id, call)
                upstream.append({"role": "tool",
                                 "tool_call_id": call.get("id", ""),
                                 "content": result})
            log.info("Active Retrieval: %d Tool-Aufruf(e) aufgelöst (Runde %d)",
                     len(ours), rounds + 1)
            events.emit("tool_call", session=resolution.session_id,
                        calls=len(ours), round=rounds + 1)
            events.emit("main_llm_start", session=resolution.session_id,
                        upstream_tokens=counter.count_messages(upstream))
            response = await llm.complete(model, upstream, **call_kwargs)
            events.emit("main_llm_done", session=resolution.session_id)
            rounds += 1

        content = response["choices"][0]["message"].get("content") or ""
        if use_tool and not content:
            # Modell hat alle Tool-Runden mit weiteren Tool-Aufrufen verbraucht
            # (LongMemEval-Befund 2026-07-18): finale Antwort ohne Tool-Angebot
            # erzwingen, sonst geht eine leere Antwort an den Client.
            log.info("Active Retrieval: Runden erschöpft ohne Antwort -> "
                     "finaler Call ohne Tools")
            upstream = upstream + [{
                "role": "user",
                "content": "Bitte beantworte die ursprüngliche Frage jetzt "
                           "direkt auf Basis der vorliegenden Informationen, "
                           "ohne weitere Tool-Aufrufe.",
            }]
            events.emit("main_llm_start", session=resolution.session_id,
                        upstream_tokens=counter.count_messages(upstream))
            response = await llm.complete(model, upstream, **kwargs)
            events.emit("main_llm_done", session=resolution.session_id)
            content = response["choices"][0]["message"].get("content") or ""
        finalize_turn(resolution.session_id, messages, content)
        return JSONResponse(content=response, headers=headers)

    @app.get("/v1/models")
    async def list_models():
        # Bei freier Modellwahl: Liste vom Upstream (z. B. LiteLLM-Proxy) holen
        if cfg.main_llm.allow_client_model and cfg.main_llm.api_base:
            try:
                import httpx

                headers = {}
                if cfg.main_llm.api_key:
                    headers["Authorization"] = f"Bearer {cfg.main_llm.api_key}"
                async with httpx.AsyncClient(timeout=5) as hc:
                    resp = await hc.get(
                        cfg.main_llm.api_base.rstrip("/") + "/models",
                        headers=headers)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                log.warning("Upstream-Modellliste nicht erreichbar - Fallback")
        return {"object": "list",
                "data": [{"id": cfg.main_llm.model, "object": "model",
                          "owned_by": "semanticswap"}]}

    # --- Admin-API (read-only, Basis für die optionale GUI - ADR-006) ---------

    @app.get("/admin/stats")
    async def admin_stats():
        return store.stats()

    @app.get("/admin/sessions")
    async def admin_sessions():
        return [
            {"id": s.id, "created_at": s.created_at, "last_active": s.last_active,
             "archived_upto": s.archived_upto, "compressing": s.compressing,
             "forked_from": s.forked_from, "messages": store.message_count(s.id)}
            for s in store.list_sessions()
        ]

    @app.get("/admin/sessions/{session_id}")
    async def admin_session_detail(session_id: str):
        session = store.get_session(session_id)
        if session is None:
            return JSONResponse(status_code=404, content={"error": "unknown session"})
        return {
            "id": session.id,
            "memory_id": session.memory_id,
            "archived_upto": session.archived_upto,
            "archive_prompt": session.archive_prompt,
            "compressing": session.compressing,
            "forked_from": session.forked_from,
            "messages": store.message_count(session.id),
            "segments": [
                {"id": seg.id, "start_idx": seg.start_idx, "end_idx": seg.end_idx,
                 "summary": seg.summary}
                for seg in store.get_segments(session.memory_id)
            ],
            "triples": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in store.get_triples(session.memory_id)
            ],
        }

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(f"{cfg.gateway.root_path}/ui/studio")

    @app.get("/v1", include_in_schema=False)
    async def v1_info():
        return {
            "service": "SemanticSwap Proxy",
            "hint": "Dies ist die Basis-URL für OpenAI-kompatible API-Clients, "
                    "keine Webseite. Trage 'http://127.0.0.1:8080/v1' in deinem "
                    "Chat-Client ein.",
            "api": {"chat": "POST /v1/chat/completions", "models": "GET /v1/models"},
            "gui": {"dashboard": "/ui", "live_flow": "/ui/flow"},
        }

    # Optionale GUI (Phase 4, ADR-006) - reiner Anzeige-Layer über dem Store
    from .gui import create_gui_router

    # Anzeigename des Standard-Modells ohne den Provider-Präfix, den der
    # Client ohnehin nie sieht (z. B. "openai/gemma4:26b" -> "gemma4:26b").
    default_model = cfg.main_llm.model
    if cfg.main_llm.client_model_prefix:
        default_model = default_model.removeprefix(cfg.main_llm.client_model_prefix)
    app.include_router(create_gui_router(store, events,
                                         prefix=cfg.gateway.root_path,
                                         default_model=default_model))

    return app

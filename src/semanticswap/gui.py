"""Optionale Web-GUI (Phase 4, ADR-006): Server-Side-Rendering unter /ui.

Read-only-Dashboard über den Systemzustand (Sessions, Archiv, Segmente, Graph).
Der Kern bleibt headless: Diese Routen sind ein dünner Anzeige-Layer über dem
Store; ohne sie funktioniert der Proxy zu 100 %. HTMX (CDN) verbessert nur die
UX (Auto-Refresh) - ohne JavaScript bleiben alle Seiten voll benutzbar.
"""
from __future__ import annotations

import asyncio
import json
from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from .events import EventBus
from .memory.store import Store

_PAGE = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>SemanticSwap - {title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>/* vor dem ersten Paint, sonst blitzt das falsche Theme auf */
document.documentElement.dataset.theme =
  localStorage.getItem('ss-theme') || 'dark';</script>
<script src="https://unpkg.com/htmx.org@1.9.12" defer></script>
<style>
  :root[data-theme="dark"] {{ color-scheme: dark;
    --bg: #0d1117; --panel: #161b22; --panel2: #21283b; --fg: #e6edf3;
    --muted: #8b949e; --border: #30363d; --accent: #2f81f7;
    --shadow: 0 4px 14px rgba(0, 0, 0, .4); }}
  :root[data-theme="solarized"] {{ color-scheme: light;
    --bg: #fdf6e3; --panel: #eee8d5; --panel2: #f7f0dc; --fg: #586e75;
    --muted: #93a1a1; --border: #d6cdb2; --accent: #268bd2;
    --shadow: 0 4px 14px rgba(88, 110, 117, .18); }}
  body {{ font-family: system-ui, sans-serif; max-width: 64rem; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5;
         background: var(--bg); color: var(--fg); }}
  h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .35rem .6rem;
            border-bottom: 1px solid var(--border); }}
  code, pre {{ background: var(--panel); border-radius: 4px; padding: .1rem .3rem; }}
  pre {{ padding: .8rem; overflow-x: auto; white-space: pre-wrap; }}
  .stats {{ display: flex; gap: .8rem; flex-wrap: wrap; }}
  .stat {{ border: 1px solid var(--border); background: var(--panel);
           border-radius: 10px; padding: .6rem 1rem; min-width: 7rem;
           cursor: help; transition: border-color .15s; }}
  .stat:hover {{ border-color: var(--accent); }}
  .stat b {{ font-size: 1.4rem; display: block; color: var(--accent); }}
  a {{ color: inherit; }}
  .muted {{ color: var(--muted); font-size: .9rem; }}
  nav {{ display: flex; align-items: center; gap: .35rem; flex-wrap: wrap; }}
  nav a {{ color: var(--muted); }}
  #themeToggle {{ margin-left: auto; padding: .3rem .7rem; border-radius: 8px;
                  border: 1px solid var(--border); background: var(--panel);
                  color: var(--fg); font: inherit; font-size: .85rem;
                  cursor: pointer; }}
  #themeToggle:hover {{ border-color: var(--accent); }}
  /* Hover-Tooltips: <element data-tip="Erklärung"> */
  [data-tip] {{ position: relative; }}
  [data-tip]:hover::after {{ content: attr(data-tip); position: absolute;
    left: 50%; bottom: calc(100% + .55rem); transform: translateX(-50%);
    background: var(--panel2); color: var(--fg);
    border: 1px solid var(--border); box-shadow: var(--shadow);
    padding: .45rem .7rem; border-radius: 8px; font-size: .78rem;
    font-weight: normal; line-height: 1.35; width: max-content;
    max-width: 16rem; white-space: normal; z-index: 10; pointer-events: none; }}
</style>
</head>
<body>
<nav class="muted"><a href="{prefix}/ui/studio"><b>SemanticSwap</b></a>
 · <a href="{prefix}/ui/studio">Studio</a> · <a href="{prefix}/ui/chat">Chat</a>
 · <a href="{prefix}/ui/flow">Live-Flow</a> · <a href="{prefix}/ui">Sessions</a>
 · <a href="{prefix}/admin/stats">Admin-API</a>
<button id="themeToggle" type="button"></button></nav>
<h1>{title}</h1>
{body}
<script>
const _tgl = document.getElementById('themeToggle');
function _tglLabel() {{
  _tgl.textContent = document.documentElement.dataset.theme === 'dark'
    ? '☀ Solarized' : '☾ Dark';
}}
_tgl.onclick = () => {{
  const next = document.documentElement.dataset.theme === 'dark'
    ? 'solarized' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('ss-theme', next);
  _tglLabel();
}};
_tglLabel();
</script>
</body>
</html>"""


def _page(title: str, body: str, prefix: str = "") -> HTMLResponse:
    return HTMLResponse(_PAGE.format(title=escape(title), prefix=prefix,
                                     body=body.replace("__PREFIX__", prefix)))


# Live-Flowchart (ADR-012). Wird NICHT durch .format() gejagt - JS-Braces sind ok.
_FLOW_BODY = """
<p class="muted">Knoten leuchten, wenn die Komponente gerade arbeitet.
Verbindung: <span id="conn">verbinde…</span></p>
<style>
  .node rect { fill: color-mix(in srgb, currentColor 6%, transparent);
               stroke: color-mix(in srgb, currentColor 45%, transparent);
               stroke-width: 1.5; rx: 10; transition: all .25s; }
  .node text { font-size: 13px; fill: currentColor; }
  .node .sub { font-size: 10px; opacity: .6; }
  .node.active rect { fill: #2f81f7; stroke: #2f81f7; }
  .node.active text { fill: #fff; }
  .edge { stroke: color-mix(in srgb, currentColor 40%, transparent);
          stroke-width: 1.5; fill: none; marker-end: url(#arr); }
  .edge.dash { stroke-dasharray: 5 4; }
  #log { list-style: none; padding: 0; font-family: ui-monospace, monospace;
         font-size: .8rem; max-height: 20rem; overflow-y: auto; }
  #log li { padding: .15rem .4rem; border-bottom: 1px solid
            color-mix(in srgb, currentColor 12%, transparent); }
  #log .t { opacity: .5; margin-right: .5rem; }
  .badge { fill: #d29922; font-size: 11px; font-weight: bold; }
</style>

<svg viewBox="0 0 760 370" style="width:100%; max-width:56rem;">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
    markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>

  <path class="edge" d="M 150 55 H 185"/>
  <path class="edge" d="M 320 55 H 355"/>
  <path class="edge" d="M 490 55 H 545"/>
  <path class="edge" d="M 255 80 V 145"/>
  <path class="edge" d="M 320 175 H 355"/>
  <path class="edge" d="M 490 175 H 545"/>
  <path class="edge" d="M 615 200 V 265"/>
  <path class="edge dash" d="M 590 285 C 725 255 700 140 655 85"/>
  <path class="edge" d="M 330 295 H 245"/>
  <path class="edge dash" d="M 140 270 C 130 170 175 100 248 82"/>

  <g id="node-client" class="node"><title>Dein Chat-Fenster oder ein
API-Client - spricht normales OpenAI-Protokoll mit dem Proxy.</title>
    <rect x="20" y="30" width="130" height="50"/>
    <text x="85" y="52" text-anchor="middle">Client / Harness</text>
    <text x="85" y="68" text-anchor="middle" class="sub">OpenAI-API</text></g>
  <g id="node-gateway" class="node"><title>Nimmt Anfragen an und ersetzt
bereits archivierte Nachrichten durch den kompakten ARCHIVE-Prompt
(virtuelles Pruning).</title>
    <rect x="190" y="30" width="130" height="50"/>
    <text x="255" y="52" text-anchor="middle">Gateway</text>
    <text x="255" y="68" text-anchor="middle" class="sub">virtuelles Pruning</text></g>
  <g id="node-tracker" class="node"><title>Erkennt Sessions über eine
Hash-Kette der Nachrichten wieder - auch Forks desselben Verlaufs.</title>
    <rect x="360" y="30" width="130" height="50"/>
    <text x="425" y="52" text-anchor="middle">Session-Tracker</text>
    <text x="425" y="68" text-anchor="middle" class="sub">Hash-Kette</text></g>
  <g id="node-mainllm" class="node"><title>Das antwortende Sprachmodell
(Standard aus der Config oder deine Dropdown-Wahl).</title>
    <rect x="550" y="30" width="130" height="50"/>
    <text x="615" y="52" text-anchor="middle">Haupt-LLM</text>
    <text x="615" y="68" text-anchor="middle" class="sub" id="mainllm-sub">idle</text></g>
  <g id="node-monitor" class="node"><title>Misst nach jeder Antwort die
Kontext-Füllung; über der Schwelle wird die Archivierung angestoßen.</title>
    <rect x="190" y="150" width="130" height="50"/>
    <text x="255" y="172" text-anchor="middle">Context Monitor</text>
    <text x="255" y="188" text-anchor="middle" class="sub" id="monitor-sub">-</text></g>
  <g id="node-queue" class="node"><title>Archivierungs-Jobs laufen asynchron
im Hintergrund - der Chat wird dadurch nicht blockiert.</title>
    <rect x="360" y="150" width="130" height="50"/>
    <text x="425" y="172" text-anchor="middle">Job-Queue</text>
    <text x="425" y="188" text-anchor="middle" class="sub">asynchron</text></g>
  <g id="node-workers" class="node"><title>Kleine LLM-Arbeiter: fassen
Abschnitte zusammen, extrahieren Fakten (Tripel) und vergeben
Prioritäten.</title>
    <rect x="550" y="150" width="130" height="50"/>
    <text x="615" y="172" text-anchor="middle">Sub-Agenten</text>
    <text x="615" y="188" text-anchor="middle" class="sub">aktiv: <tspan
      id="workerCount" class="badge">0</tspan></text></g>
  <g id="node-memory" class="node"><title>Langzeitgedächtnis in SQLite:
Zusammenfassungen, Wissens-Graph, Vektoren und Volltext-Index (FTS5).</title>
    <rect x="330" y="270" width="260" height="50"/>
    <text x="460" y="292" text-anchor="middle">Semantic Memory</text>
    <text x="460" y="308" text-anchor="middle" class="sub">SQLite · Graph · Vektoren · FTS</text></g>
  <g id="node-retrieval" class="node"><title>Retrieval-Pipeline für Swap-In und
das Retrieval-Tool: (1) Hybrid-Suche — Vektor-Ähnlichkeit plus FTS5-Volltext
(exakte Namen, Zahlen, Codes), verschmolzen per Reciprocal Rank Fusion;
(2) Re-Ranker — bewertet Query+Kandidat gemeinsam per LLM und sortiert neu
(gegen Beifang). Liefert Original-Snippets zurück in den Kontext.</title>
    <rect x="40" y="270" width="200" height="50"/>
    <text x="140" y="290" text-anchor="middle">Retrieval</text>
    <text x="140" y="306" text-anchor="middle" class="sub">Hybrid · RRF · Re-Rank</text></g>
</svg>

<h2>Ereignisse</h2>
<ul id="log"></ul>

<script>
const nodeFor = {
  request: 'gateway', session: 'tracker', swap_in: 'retrieval',
  retrieval_search: 'retrieval', graph_expansion: 'retrieval',
  rerank: 'retrieval', query_decompose: 'retrieval', tool_call: 'retrieval',
  main_llm_start: 'mainllm', main_llm_done: 'mainllm', monitor: 'monitor',
  compression_enqueued: 'queue', compression_start: 'queue',
  subagent_start: 'workers', subagent_done: 'workers', archive_updated: 'memory'
};
const labels = {
  request: 'Request eingegangen', session: 'Session zugeordnet',
  swap_in: 'Swap-In (Snippets injiziert)',
  retrieval_search: 'Speicher-Suche (Vektor + Volltext)',
  graph_expansion: 'Graph-Expansion (Nachbar-Segmente)',
  rerank: 'Re-Ranker (Kandidaten neu sortiert)',
  query_decompose: 'Query-Decomposition (Teilfragen)',
  tool_call: 'Active Retrieval (Tool)',
  main_llm_start: 'Haupt-LLM arbeitet', main_llm_done: 'Haupt-LLM fertig',
  monitor: 'Token-Check', compression_enqueued: 'Kompression eingeplant',
  compression_start: 'Kompression startet', subagent_start: 'Sub-Agent startet',
  subagent_done: 'Sub-Agent fertig', archive_updated: 'Archiv aktualisiert'
};
let workers = 0, seen = 0;
function pulse(id, sticky) {
  const el = document.getElementById('node-' + id);
  if (!el) return;
  el.classList.add('active');
  clearTimeout(el._t);
  if (!sticky) el._t = setTimeout(() => el.classList.remove('active'), 1400);
}
function release(id) {
  const el = document.getElementById('node-' + id);
  if (el) { clearTimeout(el._t); el.classList.remove('active'); }
}
function addLog(ev) {
  const li = document.createElement('li');
  const t = new Date(ev.ts * 1000).toLocaleTimeString();
  const detail = Object.entries(ev).filter(([k]) =>
    !['id','ts','type'].includes(k)).map(([k,v]) => k + '=' + v).join(' ');
  li.innerHTML = '<span class="t">' + t + '</span><b>' +
    (labels[ev.type] || ev.type) + '</b> <span class="t">' + detail + '</span>';
  const log = document.getElementById('log');
  log.prepend(li);
  while (log.children.length > 120) log.removeChild(log.lastChild);
}
// Nicht auf den eingebauten Retry verlassen: nach einem fatalen Fehler
// (z. B. 502 vom Edge-Proxy, während der Server neu startet) gibt
// EventSource endgültig auf - deshalb selbst schließen und neu verbinden.
let es;
function connect() {
  es = new EventSource('__PREFIX__/ui/events');
  es.onopen = () => document.getElementById('conn').textContent = 'live';
  es.onerror = () => {
    document.getElementById('conn').textContent = 'getrennt - reconnect…';
    es.close();
    setTimeout(connect, 4000);
  };
  es.onmessage = onEvent;
}
connect();
function onEvent(e) {
  const ev = JSON.parse(e.data);
  const isReplay = ev.id <= seen || (seen === 0 && (Date.now()/1000 - ev.ts) > 5);
  seen = Math.max(seen, ev.id);
  addLog(ev);
  if (isReplay) return;  // Replay nur ins Log, nicht blinken
  if (ev.type === 'main_llm_start') {
    pulse('mainllm', true);
    document.getElementById('mainllm-sub').textContent =
      ev.upstream_tokens + ' Token';
  }
  if (ev.type === 'main_llm_done') {
    release('mainllm');
    document.getElementById('mainllm-sub').textContent = 'idle';
  }
  if (ev.type === 'monitor') {
    document.getElementById('monitor-sub').textContent =
      ev.tokens + ' Tok (' + Math.round(ev.ratio * 100) + '%)' +
      (ev.triggered ? ' TRIGGER' : '');
  }
  if (ev.type === 'subagent_start') { workers++; pulse('workers', true); }
  if (ev.type === 'subagent_done') {
    workers = Math.max(0, workers - 1);
    if (workers === 0) release('workers');
  }
  document.getElementById('workerCount').textContent = workers;
  const node = nodeFor[ev.type];
  if (node && !['main_llm_start','main_llm_done','subagent_start','subagent_done']
      .includes(ev.type)) pulse(node);
}
</script>
"""


# Minimaler Test-Chat: verhält sich wie ein echter stateless OpenAI-Client
# (schickt immer den vollen Verlauf) - ideal, um den Proxy live zu erleben.
_CHAT_BODY = """
<p class="muted">Dieser Chat ist ein normaler API-Client des Proxys - öffne
parallel den <a href="__PREFIX__/ui/flow" target="_blank">Live-Flow</a>, um dem
Gedächtnis bei der Arbeit zuzusehen. Erste Antwort kann dauern
(Modell wird geladen).</p>
<style>
  #chat { display: flex; flex-direction: column; gap: .6rem; min-height: 18rem;
          max-height: 55vh; overflow-y: auto; padding: .8rem;
          background: var(--panel); border: 1px solid var(--border);
          border-radius: 12px; box-shadow: var(--shadow); }
  .msg { max-width: 85%; padding: .5rem .8rem; border-radius: 12px;
         white-space: pre-wrap; }
  .user { align-self: flex-end; background: var(--accent); color: #fff; }
  .assistant { align-self: flex-start; background: var(--bg);
               border: 1px solid var(--border); }
  .assistant.pending { opacity: .55; font-style: italic; }
  #bar { display: flex; gap: .5rem; margin-top: .8rem; }
  #inp { flex: 1; padding: .6rem; border-radius: 8px;
         border: 1px solid var(--border);
         background: var(--panel); color: inherit; font: inherit; }
  #inp:focus, #model:focus { outline: none; border-color: var(--accent); }
  #model { max-width: 12rem; padding: .4rem; border-radius: 8px;
           border: 1px solid var(--border);
           background: var(--panel); color: inherit; font: inherit;
           cursor: pointer; }
  /* Chrome rendert das Popup hell, sobald das select ein eigenes background
     hat - ohne explizite Optionsfarben steht dann Weiß auf Weiß (Dark-Mode).
     Canvas/CanvasText folgen dem aktiven Farbschema (color-scheme je Theme). */
  #model option { background: Canvas; color: CanvasText; }
  button { padding: .6rem 1.2rem; border-radius: 8px; border: none;
           background: var(--accent); color: #fff; font: inherit;
           cursor: pointer; }
  button:disabled { opacity: .5; }
</style>
<div id="chat"></div>
<div id="bar">
  <select id="model" title="Modell für diese Anfrage">
    <option value="">__MODEL_DEFAULT_OPTION__</option>
  </select>
  <input id="inp" placeholder="Nachricht… (Enter zum Senden)" autofocus>
  <button id="send">Senden</button>
</div>
<p class="muted">Session: <code id="sid"></code> ·
<a href="#" id="detail">Speicher dieser Session ansehen</a></p>
<script>
const sid = 'chat-' + Math.random().toString(36).slice(2, 10);
document.getElementById('sid').textContent = sid;
document.getElementById('detail').href = '__PREFIX__/ui/sessions/' + sid;
const history = [{role: 'system',
  content: 'Du bist ein hilfreicher Assistent mit Langzeitgedächtnis. Antworte auf Deutsch.'}];
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const btn = document.getElementById('send');
const modelSel = document.getElementById('model');
fetch('__PREFIX__/v1/models').then(r => r.json()).then(d => {
  (d.data || []).forEach(m => {
    const o = document.createElement('option');
    o.value = m.id; o.textContent = m.id;
    modelSel.appendChild(o);
  });
}).catch(() => {});

function bubble(role, text, pending) {
  const d = document.createElement('div');
  d.className = 'msg ' + role + (pending ? ' pending' : '');
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}
async function send() {
  const text = inp.value.trim();
  if (!text) return;
  inp.value = ''; btn.disabled = true; inp.disabled = true;
  history.push({role: 'user', content: text});
  bubble('user', text);
  const pending = bubble('assistant', 'denkt nach…', true);
  try {
    // stream: true - die Antwort kommt in Häppchen. Wichtig hinter
    // Edge-Proxys (Vercel kappt stille Verbindungen nach ~60 s).
    const resp = await fetch('__PREFIX__/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'x-session-id': sid},
      body: JSON.stringify({model: modelSel.value || 'semanticswap',
                            messages: history, stream: true})
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '', raw = '';
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, {stream: true});
      const parts = buf.split('\\n\\n'); buf = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;   // ignoriert ':'-Keepalives
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        let delta = '';
        try { delta = JSON.parse(payload).choices?.[0]?.delta?.content || ''; }
        catch (e) { continue; }
        if (!delta) continue;
        raw += delta;
        // Reasoning-Modelle: <think>-Phase nicht anzeigen
        let shown = raw;
        if (raw.includes('</think>')) shown = raw.split('</think>').pop();
        else if (raw.includes('<think>')) shown = '';
        shown = shown.trimStart();
        if (shown) {
          pending.textContent = shown;
          pending.classList.remove('pending');
          chat.scrollTop = chat.scrollHeight;
        }
      }
    }
    if (!raw) throw new Error('leere Antwort');
    history.push({role: 'assistant', content: raw});
    const answer = raw.includes('</think>')
      ? raw.split('</think>').pop().trim() : raw.trim();
    pending.textContent = answer; pending.classList.remove('pending');
  } catch (err) {
    pending.textContent = 'Fehler: ' + err.message;
    pending.classList.remove('pending');
  } finally {
    btn.disabled = false; inp.disabled = false; inp.focus();
  }
}
btn.onclick = send;
inp.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
"""


def _stats_cards(store: Store, prefix: str = "") -> str:
    stats = store.stats()

    def _de(value: float, suffix: str) -> str:
        return f"{value:.1f}".replace(".", ",") + suffix

    # Auswertung: Zeichenlängen als Token-Proxy (der Store speichert keine
    # Token-Zahlen); "-" solange noch nichts archiviert wurde.
    ratio = (_de(stats["archived_chars"] / stats["summary_chars"], ":1")
             if stats["summary_chars"] else "–")
    savings = (_de(stats["archived_chars"] / stats["prompt_chars"], "x")
               if stats["prompt_chars"] else "–")

    tiles: list[tuple[str, str, str]] = [
        (str(stats["sessions"]), "Sessions",
         "Getrackte Gespräche (Hash-Ketten), inkl. Forks und Test-Sessions"),
        (str(stats["messages"]), "Messages",
         "Gespeicherte Original-Nachrichten über alle Sessions"),
        (str(stats["segments"]), "Segmente",
         "Komprimierte Gesprächsabschnitte im Archiv (Swap-Out)"),
        (str(stats["triples"]), "Tripel",
         "Extrahierte Wissens-Fakten: Subjekt, Prädikat, Objekt"),
        (str(stats["embeddings"]), "Embeddings",
         "Vektoren für die semantische Suche (Swap-In)"),
        (ratio, "Archiv-Ratio",
         "Originaltext zu Zusammenfassungen (Zeichen) - je höher, desto "
         "stärker verdichtet das Archiv"),
        (savings, "Kontext-Ersparnis",
         "Archivierter Originaltext zu dem, was stattdessen im Kontext "
         "landet (ARCHIVE-Prompt, Zeichen)"),
        (str(stats["low_priority_segments"]), "Deep Archive",
         "Low-Priority-Segmente: ältere verschwinden aus dem Prompt, "
         "bleiben aber per Retrieval abrufbar"),
    ]
    cards = "".join(
        f'<div class="stat" data-tip="{escape(tip)}"><b>{escape(value)}</b>'
        f"{escape(label)}</div>"
        for value, label, tip in tiles
    )
    # hx-Attribute: Kacheln aktualisieren sich alle 5 s selbst (Fallback: statisch)
    return (f'<div class="stats" hx-get="{prefix}/ui" hx-trigger="every 5s" '
            f'hx-select=".stats" hx-swap="outerHTML">{cards}</div>')


def create_gui_router(store: Store, bus: EventBus, prefix: str = "",
                      default_model: str | None = None) -> APIRouter:
    router = APIRouter()

    # Erste Dropdown-Option zeigt das konfigurierte Standard-Modell.
    default_label = (f"Modell: {default_model} (Standard)" if default_model
                     else "Modell: Standard")
    chat_body = _CHAT_BODY.replace("__MODEL_DEFAULT_OPTION__",
                                   escape(default_label))

    @router.get("/ui/studio", response_class=HTMLResponse)
    async def studio():
        body = (
            "<style>"
            "body { max-width: 110rem; }"
            ".studio { display: grid; gap: 2rem; grid-template-columns: 1fr; "
            "          margin-top: 1rem; }"
            "@media (min-width: 1100px) { .studio { grid-template-columns: "
            "  minmax(24rem, 42rem) 1fr; align-items: start; } }"
            ".pane h2 { margin-top: 0; }"
            "</style>"
            + _stats_cards(store, prefix)
            + '<div class="studio">'
            + f'<section class="pane"><h2>Chat</h2>{chat_body}</section>'
            + f'<section class="pane"><h2>Live-Flow</h2>{_FLOW_BODY}</section>'
            + "</div>"
        )
        return _page("Studio", body, prefix)

    @router.get("/ui/events")
    async def event_stream(replay_only: bool = False):
        """SSE-Stream (ADR-012): Replay der letzten Events + Live-Feed.
        replay_only beendet nach dem Replay (für Tests/Debugging)."""
        async def gen():
            queue = bus.subscribe()
            try:
                for event in bus.history:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if replay_only:
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.get("/ui/flow", response_class=HTMLResponse)
    async def flow():
        return _page("Live-Flow", _FLOW_BODY, prefix)

    @router.get("/ui/chat", response_class=HTMLResponse)
    async def chat():
        return _page("Chat", chat_body, prefix)

    @router.get("/ui", response_class=HTMLResponse)
    async def dashboard():
        rows = []
        for s in store.list_sessions():
            rows.append(
                f"<tr><td><a href='{prefix}/ui/sessions/{escape(s.id)}'>"
                f"<code>{escape(s.id)}</code></a></td>"
                f"<td>{store.message_count(s.id)}</td>"
                f"<td>{s.archived_upto}</td>"
                f"<td>{'ja' if s.compressing else '-'}</td>"
                f"<td><code>{escape(s.forked_from or '-')}</code></td>"
                f"<td class='muted'>{escape(s.last_active[:19])}</td></tr>"
            )
        body = (
            _stats_cards(store)
            + f"<h2>Sessions</h2>"
            f"<table><tr><th>ID</th><th>Messages</th><th>archiviert bis</th>"
            f"<th>komprimiert grade</th><th>Fork von</th><th>zuletzt aktiv</th></tr>"
            f"{''.join(rows) or '<tr><td colspan=6 class=muted>noch keine</td></tr>'}"
            f"</table>"
        )
        return _page("Dashboard", body, prefix)

    @router.get("/ui/sessions/{session_id}", response_class=HTMLResponse)
    async def session_detail(session_id: str):
        session = store.get_session(session_id)
        if session is None:
            return _page("Unbekannte Session",
                         "<p>Diese Session existiert nicht (mehr).</p>", prefix)
        segments = store.get_segments(session.memory_id)
        triples = store.get_triples(session.memory_id)
        seg_rows = "".join(
            f"<tr><td><code>{escape(seg.id)}</code></td>"
            f"<td>{seg.start_idx}-{seg.end_idx}</td>"
            f"<td>{escape(seg.summary[:220])}</td></tr>"
            for seg in segments
        )
        triple_rows = "".join(
            f"<tr><td>{escape(s)}</td><td>{escape(p)}</td><td>{escape(o)}</td></tr>"
            for s, p, o in triples[:200]
        )
        body = (
            f"<p>Speicherraum <code>{escape(session.memory_id)}</code> · "
            f"{store.message_count(session.id)} Messages · archiviert bis "
            f"Message {session.archived_upto}"
            + (f" · Fork von <code>{escape(session.forked_from)}</code>"
               if session.forked_from else "")
            + "</p>"
            f"<h2>ARCHIVE-Prompt</h2><pre>{escape(session.archive_prompt or '(noch keiner)')}</pre>"
            f"<h2>Segmente ({len(segments)})</h2>"
            f"<table><tr><th>ID</th><th>Messages</th><th>Summary</th></tr>"
            f"{seg_rows or '<tr><td colspan=3 class=muted>keine</td></tr>'}</table>"
            f"<h2>Graph-Tripel ({len(triples)})</h2>"
            f"<table><tr><th>Subjekt</th><th>Prädikat</th><th>Objekt</th></tr>"
            f"{triple_rows or '<tr><td colspan=3 class=muted>keine</td></tr>'}</table>"
        )
        return _page(f"Session {session.id}", body, prefix)

    return router

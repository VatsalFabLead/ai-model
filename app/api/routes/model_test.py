"""Model test page — local/live diagnostics for the custom model."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import get_settings

router = APIRouter(tags=["pages"])


@router.get("/model_test", response_class=HTMLResponse)
async def model_test_page() -> str:
  settings = get_settings()
  return _HTML.format(
    app_name=settings.app_name,
    api_prefix=settings.api_prefix,
    model_backend=settings.model_backend,
    model_id=settings.model_id,
    app_env=settings.app_env,
  )


@router.get("/model_test/status")
async def model_test_status(request: Request) -> JSONResponse:
  """JSON diagnostics for scripts or the test page."""
  settings = get_settings()
  registry = request.app.state.registry
  ready = registry.is_ready()
  model_id = registry.provider.model_id() if ready else None
  return JSONResponse(
    {
      "service": settings.app_name,
      "environment": settings.app_env,
      "model_backend": settings.model_backend,
      "model_ready": ready,
      "model_id": model_id,
      "api_prefix": settings.api_prefix,
      "health_url": "/health",
      "chat_url": f"{settings.api_prefix}/chat/completions",
      "models_url": f"{settings.api_prefix}/models",
    }
  )


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{app_name} — Model Test</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3b82f6;
      --ok: #22c55e;
      --warn: #f59e0b;
      --err: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .wrap {{ max-width: 720px; margin: 0 auto; padding: 2rem 1.25rem; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; margin: 0 0 0.25rem; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1rem;
    }}
    .row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
    label {{ font-size: 0.8rem; color: var(--muted); display: block; margin-bottom: 0.35rem; }}
    input, textarea {{
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      padding: 0.6rem 0.75rem;
      font-size: 0.95rem;
    }}
    .key-row {{ display: flex; gap: 0.5rem; align-items: stretch; }}
    .key-row input {{ flex: 1; }}
    .icon-btn {{
      background: var(--border);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0 0.85rem;
      cursor: pointer;
      font-size: 1rem;
      line-height: 1;
    }}
    .icon-btn:hover {{ filter: brightness(1.1); }}
    textarea {{ min-height: 80px; resize: vertical; font-family: inherit; }}
    button {{
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.6rem 1.1rem;
      font-size: 0.9rem;
      cursor: pointer;
      font-weight: 500;
    }}
    button:hover {{ filter: brightness(1.08); }}
    button.secondary {{ background: var(--border); }}
    .badge {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge.ok {{ background: rgba(34,197,94,0.2); color: var(--ok); }}
    .badge.warn {{ background: rgba(245,158,11,0.2); color: var(--warn); }}
    .badge.err {{ background: rgba(239,68,68,0.2); color: var(--err); }}
    pre {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      overflow: auto;
      font-size: 0.8rem;
      margin: 0.5rem 0 0;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .nav {{ margin-top: 1.5rem; font-size: 0.9rem; }}
    .nav a {{ color: var(--accent); text-decoration: none; }}
    .nav a:hover {{ text-decoration: underline; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.75rem; }}
    .stat {{ font-size: 0.8rem; color: var(--muted); }}
    .stat strong {{ display: block; color: var(--text); font-size: 0.95rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Model Test</h1>
    <p class="sub">{app_name} · backend <code>{model_backend}</code> · env <code>{app_env}</code></p>

    <div class="card">
      <div class="row" style="justify-content: space-between; margin-bottom: 1rem;">
        <span id="statusBadge" class="badge warn">checking…</span>
        <button type="button" id="btnRefresh" class="secondary">Refresh status</button>
      </div>
      <div class="grid" id="stats"></div>
    </div>

    <div class="card">
      <label for="apiKey">API key (Bearer)</label>
      <div class="key-row">
        <input id="apiKey" type="password" placeholder="change-me-to-a-strong-key" autocomplete="off" />
        <button type="button" id="btnToggleKey" class="icon-btn" title="Show/hide API key" aria-label="Show API key">👁</button>
      </div>
      <p style="font-size:0.8rem;color:var(--muted);margin:0.5rem 0 0;">Saved in sessionStorage for this tab only. Powers all tools via <code>{model_id}</code>.</p>
    </div>

    <div class="card">
      <label for="testPrompt">Quick chat test</label>
      <textarea id="testPrompt" placeholder="Say hello and introduce yourself.">Hello! Who are you?</textarea>
      <div class="row" style="margin-top: 0.75rem;">
        <button type="button" id="btnRun">Run inference</button>
      </div>
      <pre id="testOut">Response will appear here…</pre>
    </div>

    <p class="nav">
      <a href="/chat_page">Open chat page →</a>
      &nbsp;·&nbsp;
      <a href="/docs">API docs</a>
      &nbsp;·&nbsp;
      <a href="/health" target="_blank">/health</a>
    </p>
  </div>
  <script>
    const API_PREFIX = "{api_prefix}";
    const MODEL_ID = "{model_id}";
    const keyInput = document.getElementById("apiKey");
    const saved = sessionStorage.getItem("api_key");
    if (saved) keyInput.value = saved;
    keyInput.addEventListener("change", () => sessionStorage.setItem("api_key", keyInput.value));
    document.getElementById("btnToggleKey").onclick = () => {{
      const show = keyInput.type === "password";
      keyInput.type = show ? "text" : "password";
      document.getElementById("btnToggleKey").setAttribute("aria-label", show ? "Hide API key" : "Show API key");
    }};

    function headers() {{
      const h = {{ "Content-Type": "application/json" }};
      const k = keyInput.value.trim();
      if (k) h["Authorization"] = "Bearer " + k;
      return h;
    }}

    async function loadStatus() {{
      const badge = document.getElementById("statusBadge");
      const stats = document.getElementById("stats");
      try {{
        const [health, meta] = await Promise.all([
          fetch("/health").then(r => r.json()),
          fetch("/model_test/status").then(r => r.json()),
        ]);
        const ok = health.model_ready;
        badge.textContent = ok ? "model ready" : "loading / unavailable";
        badge.className = "badge " + (ok ? "ok" : "warn");
        stats.innerHTML = [
          ["Model ID", meta.model_id || MODEL_ID || "—"],
          ["Backend", meta.model_backend],
          ["Environment", meta.environment],
          ["Health", health.status],
        ].map(([l, v]) => `<div class="stat"><span>${{l}}</span><strong>${{v}}</strong></div>`).join("");
      }} catch (e) {{
        badge.textContent = "error";
        badge.className = "badge err";
        stats.innerHTML = `<pre>${{e.message}}</pre>`;
      }}
    }}

    document.getElementById("btnRefresh").onclick = loadStatus;

    document.getElementById("btnRun").onclick = async () => {{
      const out = document.getElementById("testOut");
      out.textContent = "Running…";
      const prompt = document.getElementById("testPrompt").value.trim();
      try {{
        const res = await fetch(API_PREFIX + "/chat/completions", {{
          method: "POST",
          headers: headers(),
          body: JSON.stringify({{
            model: MODEL_ID,
            messages: [{{ role: "user", content: prompt }}],
            max_tokens: 256,
            temperature: 0.7,
          }}),
        }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        const text = data.choices?.[0]?.message?.content || JSON.stringify(data, null, 2);
        out.textContent = text;
      }} catch (e) {{
        out.textContent = "Error: " + e.message;
      }}
    }};

    loadStatus();
  </script>
</body>
</html>
"""

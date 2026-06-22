"""Chat page — browser UI for testing chat on local and live."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import get_settings

router = APIRouter(tags=["pages"])


@router.get("/chat_page", response_class=HTMLResponse)
async def chat_page() -> str:
  settings = get_settings()
  return _HTML.format(
    app_name=settings.app_name,
    api_prefix=settings.api_prefix,
    model_backend=settings.model_backend,
  )


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{app_name} — Chat</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --user: #1e3a5f;
      --bot: #1a2332;
      --accent: #3b82f6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 0.85rem 1.25rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    header h1 {{ font-size: 1.1rem; margin: 0; font-weight: 600; }}
    header .meta {{ font-size: 0.75rem; color: var(--muted); }}
    header a {{ color: var(--accent); text-decoration: none; font-size: 0.85rem; }}
    .settings {{
      padding: 0.6rem 1.25rem;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      align-items: flex-end;
    }}
    .settings label {{ font-size: 0.7rem; color: var(--muted); display: block; margin-bottom: 0.25rem; }}
    .settings input {{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      padding: 0.4rem 0.6rem;
      font-size: 0.85rem;
      width: 160px;
    }}
    #messages {{
      flex: 1;
      overflow-y: auto;
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}
    .msg {{
      max-width: 85%;
      padding: 0.65rem 0.9rem;
      border-radius: 12px;
      font-size: 0.95rem;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .msg.user {{
      align-self: flex-end;
      background: var(--user);
      border-bottom-right-radius: 4px;
    }}
    .msg.assistant {{
      align-self: flex-start;
      background: var(--bot);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }}
    .msg.system {{
      align-self: center;
      font-size: 0.8rem;
      color: var(--muted);
      background: transparent;
      padding: 0.25rem;
    }}
    .composer {{
      padding: 0.85rem 1.25rem;
      border-top: 1px solid var(--border);
      display: flex;
      gap: 0.6rem;
      align-items: flex-end;
    }}
    .composer textarea {{
      flex: 1;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      padding: 0.65rem 0.85rem;
      font-size: 0.95rem;
      font-family: inherit;
      resize: none;
      min-height: 44px;
      max-height: 120px;
    }}
    .composer button {{
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0.65rem 1.2rem;
      font-weight: 500;
      cursor: pointer;
    }}
    .composer button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .typing {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>{app_name}</h1>
      <div class="meta">Custom model · {model_backend}</div>
    </div>
    <a href="/model_test">Model test</a>
  </header>

  <div class="settings">
    <div>
      <label>API key</label>
      <input id="apiKey" type="password" placeholder="Bearer token" />
    </div>
    <div>
      <label>Max tokens</label>
      <input id="maxTokens" type="number" value="256" min="16" max="2048" />
    </div>
    <div>
      <label>Temperature</label>
      <input id="temperature" type="number" value="0.7" min="0" max="2" step="0.1" />
    </div>
  </div>

  <div id="messages">
    <div class="msg system">Send a message to test your custom model locally or on live.</div>
  </div>

  <div class="composer">
    <textarea id="input" rows="1" placeholder="Type a message…" autofocus></textarea>
    <button type="button" id="send">Send</button>
  </div>

  <script>
    const API_PREFIX = "{api_prefix}";
    const messagesEl = document.getElementById("messages");
    const inputEl = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    const keyEl = document.getElementById("apiKey");

    const history = [];
    const savedKey = sessionStorage.getItem("api_key");
    if (savedKey) keyEl.value = savedKey;
    keyEl.addEventListener("change", () => sessionStorage.setItem("api_key", keyEl.value));

    function addMsg(role, text) {{
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }}

    function headers() {{
      const h = {{ "Content-Type": "application/json" }};
      const k = keyEl.value.trim();
      if (k) h["Authorization"] = "Bearer " + k;
      return h;
    }}

    async function send() {{
      const text = inputEl.value.trim();
      if (!text) return;

      inputEl.value = "";
      addMsg("user", text);
      history.push({{ role: "user", content: text }});

      sendBtn.disabled = true;
      const typing = addMsg("assistant", "");
      typing.classList.add("typing");
      typing.textContent = "Thinking…";

      try {{
        const res = await fetch(API_PREFIX + "/chat/completions", {{
          method: "POST",
          headers: headers(),
          body: JSON.stringify({{
            messages: history,
            max_tokens: parseInt(document.getElementById("maxTokens").value, 10) || 256,
            temperature: parseFloat(document.getElementById("temperature").value) || 0.7,
          }}),
        }});
        const data = await res.json();
        if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
        const reply = data.choices[0].message.content;
        history.push({{ role: "assistant", content: reply }});
        typing.classList.remove("typing");
        typing.textContent = reply;
      }} catch (e) {{
        typing.classList.remove("typing");
        typing.textContent = "Error: " + e.message;
      }} finally {{
        sendBtn.disabled = false;
        inputEl.focus();
      }}
    }}

    sendBtn.onclick = send;
    inputEl.addEventListener("keydown", (e) => {{
      if (e.key === "Enter" && !e.shiftKey) {{
        e.preventDefault();
        send();
      }}
    }});
  </script>
</body>
</html>
"""

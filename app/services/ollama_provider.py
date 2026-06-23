"""Free open-source local LLM provider via Ollama.

Ollama runs OPEN-SOURCE models (Qwen, Llama, Mistral, etc.) entirely on your own
machine/server. It is NOT GPT/Claude/Gemini and involves no external AI company or
API. We talk to it over local HTTP, so it works on any Python version (no native
build needed). Optional RAG context comes from your custom KB + free Wikipedia.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.config import Settings
from app.engine.answer_complete import CONTINUE_PROMPT, is_incomplete_answer, merge_continuation
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.tool_context import ToolContextBuilder
from app.services.provider_base import ModelProvider


class OllamaProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._ready = False
    self._tools = ToolContextBuilder(settings)

  async def load(self) -> None:
    base = self._settings.ollama_host.rstrip("/")
    try:
      async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{base}/api/tags")
        r.raise_for_status()
    except httpx.HTTPError as exc:
      raise RuntimeError(
        f"Could not reach Ollama at {base}. Is it installed and running?\n"
        f"  1) Install: https://ollama.com/download\n"
        f"  2) Pull a model: ollama pull {self._settings.ollama_model}\n"
        f"Original error: {exc}"
      ) from exc

    self._tools.load_kb()
    self._ready = True

  async def unload(self) -> None:
    self._ready = False

  def is_ready(self) -> bool:
    return self._ready

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    if not kwargs.get("skip_intent") and detect_resume_intent(last_user):
      return generate_resume(last_user)

    if is_gold_price_query(last_user):
      live = await fetch_gold_price_context(last_user)
      if live:
        return live

    system_prompt = kwargs.get("system_prompt") or self._settings.llm_system_prompt
    use_rag = kwargs.get("use_rag")
    if use_rag is None:
      use_rag = self._settings.use_rag

    full_messages: list[dict[str, str]] = [
      {"role": "system", "content": system_prompt}
    ]
    if use_rag and last_user.strip():
      context = await self._tools.gather(last_user)
      if context:
        full_messages.append({
          "role": "system",
          "content": f"Reference context (use if relevant):\n{context[:28000]}",
        })
    full_messages.extend(messages)

    temperature = kwargs.get("temperature")
    temperature = float(temperature if temperature is not None else self._settings.llm_temperature)
    max_tokens = int(kwargs.get("max_tokens") or self._settings.llm_max_tokens)
    max_passes = max(1, self._settings.chat_completion_max_passes)
    base = self._settings.ollama_host.rstrip("/")

    answer = ""
    for attempt in range(max_passes):
      run_messages = list(full_messages)
      if attempt > 0 and answer.strip():
        run_messages = run_messages + [
          {"role": "assistant", "content": answer.strip()},
          {"role": "user", "content": CONTINUE_PROMPT},
        ]
      payload = {
        "model": self._settings.ollama_model,
        "messages": run_messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
      }
      async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
      chunk = (data.get("message", {}).get("content") or "").strip()
      answer = merge_continuation(answer, chunk)
      done = bool(data.get("done", True))
      if done and not is_incomplete_answer(answer):
        break

    return answer

  def model_id(self) -> str:
    return f"ollama/{self._settings.ollama_model}"

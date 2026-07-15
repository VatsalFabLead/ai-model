"""Hosted LLM provider — OpenAI-compatible API for human-quality conversation.

Works with any OpenAI-compatible endpoint. Free options:
  Groq       https://api.groq.com/openai/v1     (llama-3.3-70b-versatile)
  OpenRouter https://openrouter.ai/api/v1        (many :free models)
  Gemini     https://generativelanguage.googleapis.com/v1beta/openai
             (gemini-2.0-flash)

Configure via .env / Render environment:
  HOSTED_LLM_ENABLED=true
  HOSTED_LLM_BASE_URL=https://api.groq.com/openai/v1
  HOSTED_LLM_API_KEY=<your key>
  HOSTED_LLM_MODEL=llama-3.3-70b-versatile

Replies are branded as the custom Nexus model; the real backend is reported
only in the X-Nexus-Backend response header.
"""

from __future__ import annotations

import httpx

from app.config import Settings
from app.services.provider_base import ModelProvider


class HostedLLMProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._ready = False

  async def load(self) -> None:
    if not self._settings.hosted_llm_enabled:
      raise RuntimeError("Hosted LLM disabled — set HOSTED_LLM_ENABLED=true")
    if not self._settings.hosted_llm_api_key.strip():
      raise RuntimeError(
        "HOSTED_LLM_API_KEY is not set. Get a free key at https://console.groq.com "
        "(or OpenRouter / Google AI Studio) and add it to the environment."
      )
    self._ready = True

  async def unload(self) -> None:
    self._ready = False

  def is_ready(self) -> bool:
    return self._ready

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    has_system = any(m.get("role") == "system" for m in messages)
    full_messages: list[dict[str, str]] = []
    if not has_system:
      full_messages.append({
        "role": "system",
        "content": self._settings.hosted_system_prompt,
      })
    full_messages.extend(
      {"role": m.get("role", "user"), "content": m.get("content", "")}
      for m in messages
    )

    payload: dict = {
      "model": self._settings.hosted_llm_model,
      "messages": full_messages,
      "stream": False,
    }
    if kwargs.get("max_tokens") is not None:
      payload["max_tokens"] = int(kwargs["max_tokens"])
    if kwargs.get("temperature") is not None:
      payload["temperature"] = float(kwargs["temperature"])
    if kwargs.get("top_p") is not None:
      payload["top_p"] = float(kwargs["top_p"])

    base = self._settings.hosted_llm_base_url.rstrip("/")
    headers = {
      "Authorization": f"Bearer {self._settings.hosted_llm_api_key.strip()}",
      "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=self._settings.hosted_llm_timeout) as client:
      r = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
      r.raise_for_status()
      data = r.json()

    choices = data.get("choices") or []
    if not choices:
      raise RuntimeError(f"Hosted LLM returned no choices: {data}")
    return (choices[0].get("message", {}).get("content") or "").strip()

  def model_id(self) -> str:
    # Branded as the product model; real backend shows in X-Nexus-Backend.
    return self._settings.model_id

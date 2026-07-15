"""Hosted LLM provider — any OpenAI-compatible chat API (Groq, Gemini, OpenRouter).

Gives /chat/completions real ChatGPT-class answers for general conversation.
Configure via env:
  HOSTED_LLM_ENABLED=true
  HOSTED_LLM_BASE_URL=https://api.groq.com/openai/v1        (Groq — free tier)
    or https://generativelanguage.googleapis.com/v1beta/openai (Gemini)
    or https://openrouter.ai/api/v1                          (OpenRouter)
  HOSTED_LLM_API_KEY=<your key>
  HOSTED_LLM_MODEL=llama-3.3-70b-versatile                   (or gemini-2.5-flash, ...)
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
      raise RuntimeError("Hosted LLM is disabled (set HOSTED_LLM_ENABLED=true)")
    if not self._settings.hosted_llm_api_key:
      raise RuntimeError("HOSTED_LLM_API_KEY is not set")
    self._ready = True

  async def unload(self) -> None:
    self._ready = False

  def is_ready(self) -> bool:
    return self._ready

  def model_id(self) -> str:
    return self._settings.hosted_llm_model

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    payload_messages = list(messages)
    if not any(m.get("role") == "system" for m in payload_messages):
      payload_messages.insert(
        0,
        {"role": "system", "content": self._settings.hosted_llm_system_prompt},
      )

    body: dict = {
      "model": self._settings.hosted_llm_model,
      "messages": payload_messages,
      "max_tokens": int(kwargs.get("max_tokens") or 1024),
      "temperature": float(kwargs.get("temperature") or 0.7),
    }
    if kwargs.get("top_p") is not None:
      body["top_p"] = float(kwargs["top_p"])

    url = self._settings.hosted_llm_base_url.rstrip("/") + "/chat/completions"
    headers = {
      "Authorization": f"Bearer {self._settings.hosted_llm_api_key}",
      "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=self._settings.hosted_llm_timeout) as client:
      res = await client.post(url, json=body, headers=headers)
    if res.status_code != 200:
      detail = res.text[:300]
      raise RuntimeError(f"Hosted LLM error {res.status_code}: {detail}")
    data = res.json()
    choices = data.get("choices") or []
    if not choices:
      raise RuntimeError("Hosted LLM returned no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
      raise RuntimeError("Hosted LLM returned empty content")
    return content.strip()

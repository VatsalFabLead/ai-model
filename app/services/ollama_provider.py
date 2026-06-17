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
from app.engine.knowledge import KnowledgeBase, load_knowledge_base
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.web_knowledge import WikipediaSource
from app.services.provider_base import ModelProvider


class OllamaProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._ready = False
    self._kb: KnowledgeBase | None = None
    self._wiki: WikipediaSource | None = None
    if settings.enable_web_knowledge:
      self._wiki = WikipediaSource(
        sentences=settings.web_knowledge_sentences,
        timeout=settings.web_knowledge_timeout,
      )

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

    self._kb = load_knowledge_base(
      knowledge_path=Path(self._settings.knowledge_path),
      corpus_path=Path(self._settings.corpus_path),
    )
    self._ready = True

  async def unload(self) -> None:
    self._ready = False
    self._kb = None

  def is_ready(self) -> bool:
    return self._ready

  async def _build_context(self, query: str) -> str:
    parts: list[str] = []
    if self._kb and self._kb.size:
      answer, score = self._kb.search(query)
      if answer and score >= self._settings.knowledge_threshold:
        parts.append(answer)
    if self._wiki is not None:
      wiki = await self._wiki.query(query)
      if wiki:
        parts.append(wiki)
    return "\n\n".join(parts)[:3000]

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    if not kwargs.get("skip_intent") and detect_resume_intent(last_user):
      return generate_resume(last_user)

    system_prompt = kwargs.get("system_prompt") or self._settings.llm_system_prompt
    use_rag = kwargs.get("use_rag")
    if use_rag is None:
      use_rag = self._settings.use_rag

    full_messages: list[dict[str, str]] = [
      {"role": "system", "content": system_prompt}
    ]
    if use_rag and last_user.strip():
      context = await self._build_context(last_user)
      if context:
        full_messages.append({
          "role": "system",
          "content": f"Reference context (use if relevant):\n{context}",
        })
    full_messages.extend(messages)

    temperature = kwargs.get("temperature")
    temperature = float(temperature if temperature is not None else self._settings.llm_temperature)
    max_tokens = int(kwargs.get("max_tokens") or self._settings.llm_max_tokens)
    base = self._settings.ollama_host.rstrip("/")
    payload = {
      "model": self._settings.ollama_model,
      "messages": full_messages,
      "stream": False,
      "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    async with httpx.AsyncClient(timeout=120) as client:
      r = await client.post(f"{base}/api/chat", json=payload)
      r.raise_for_status()
      data = r.json()
    return (data.get("message", {}).get("content") or "").strip()

  def model_id(self) -> str:
    return self._settings.model_id

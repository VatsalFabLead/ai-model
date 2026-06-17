"""Free open-source local LLM provider (GGUF via llama.cpp).

This runs an OPEN-SOURCE model on your own machine/server. It is NOT GPT/Claude/
Gemini and uses no external AI company or API. Optional RAG context is drawn from
your custom knowledge base and the free Wikipedia source.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.engine.knowledge import KnowledgeBase, load_knowledge_base
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.web_knowledge import WikipediaSource
from app.services.provider_base import ModelProvider


class LocalLLMProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._llm = None
    self._kb: KnowledgeBase | None = None
    self._wiki: WikipediaSource | None = None
    if settings.enable_web_knowledge:
      self._wiki = WikipediaSource(
        sentences=settings.web_knowledge_sentences,
        timeout=settings.web_knowledge_timeout,
      )

  def _load_sync(self) -> None:
    from llama_cpp import Llama

    model_path = Path(self._settings.llm_model_path)
    if not model_path.exists():
      raise FileNotFoundError(
        f"LLM model not found at {model_path}. Download it first:\n"
        f"  python scripts/download_model.py"
      )
    self._llm = Llama(
      model_path=str(model_path),
      n_ctx=self._settings.llm_context,
      n_threads=self._settings.llm_threads,
      n_gpu_layers=self._settings.llm_gpu_layers,
      verbose=False,
    )
    self._kb = load_knowledge_base(
      knowledge_path=Path(self._settings.knowledge_path),
      corpus_path=Path(self._settings.corpus_path),
    )

  async def load(self) -> None:
    await asyncio.to_thread(self._load_sync)

  async def unload(self) -> None:
    self._llm = None
    self._kb = None

  def is_ready(self) -> bool:
    return self._llm is not None

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

  def _generate_sync(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    resp = self._llm.create_chat_completion(
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=self._settings.llm_top_p,
      top_k=self._settings.llm_top_k,
      repeat_penalty=self._settings.llm_repeat_penalty,
      frequency_penalty=self._settings.llm_frequency_penalty,
      presence_penalty=self._settings.llm_presence_penalty,
    )
    return resp["choices"][0]["message"]["content"].strip()

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    # Deterministic, no-model resume generation when the user asks for one.
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

    max_tokens = int(kwargs.get("max_tokens") or self._settings.llm_max_tokens)
    temperature = kwargs.get("temperature")
    temperature = float(temperature if temperature is not None else self._settings.llm_temperature)
    return await asyncio.to_thread(self._generate_sync, full_messages, max_tokens, temperature)

  def model_id(self) -> str:
    return self._settings.model_id

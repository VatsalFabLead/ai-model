"""Provider registry — routes chat to custom, ollama, or llm per request."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.engine.chat_intents import (
  extract_title_meta_topic,
  format_title_meta_reply,
  is_title_meta_query,
)
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.services.backend_router import VALID_BACKENDS, is_low_quality_output, resolve_backend
from app.services.custom_provider import CustomModelProvider
from app.services.provider_base import ModelProvider


class ProviderRegistry:
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._providers: dict[str, ModelProvider | None] = {}
    self._last_backend: str = settings.model_backend

  def _create_provider(self, backend: str) -> ModelProvider:
    if backend == "gemma":
      from app.services.gemma_provider import GemmaProvider

      return GemmaProvider(self._settings)
    if backend == "ollama":
      from app.services.ollama_provider import OllamaProvider

      return OllamaProvider(self._settings)
    if backend == "llm":
      from app.services.llm_provider import LocalLLMProvider

      return LocalLLMProvider(self._settings)
    return CustomModelProvider(self._settings)

  async def _ensure_loaded(self, backend: str) -> ModelProvider | None:
    if backend not in VALID_BACKENDS:
      backend = "custom"
    if backend in self._providers and self._providers[backend] is not None:
      return self._providers[backend]
    if backend in self._providers and self._providers[backend] is None:
      return None

    provider = self._create_provider(backend)
    try:
      await provider.load()
      self._providers[backend] = provider
      return provider
    except Exception:
      self._providers[backend] = None
      return None

  async def startup(self) -> None:
    default = self._settings.model_backend.lower().strip()
    await self._ensure_loaded("custom")
    if self._settings.gemma_enabled:
      await self._ensure_loaded("gemma")
    if default == "ollama":
      await self._ensure_loaded("ollama")
    elif default == "llm" and self._settings.llm_backend_enabled:
      await self._ensure_loaded("llm")

  async def shutdown(self) -> None:
    for backend, provider in list(self._providers.items()):
      if provider:
        await provider.unload()
      self._providers[backend] = None
    self._providers.clear()

  @property
  def last_backend(self) -> str:
    return self._last_backend

  @property
  def provider(self) -> ModelProvider:
    """Default provider (for health checks / legacy callers)."""
    for key in ("gemma", "custom", "ollama", "llm"):
      p = self._providers.get(key)
      if p and p.is_ready():
        return p
    raise RuntimeError("Provider registry is not initialized")

  def tool_provider(self) -> ModelProvider:
    """Best backend for AI tools: Gemma + Nexus RAG, then fallbacks."""
    for key in ("gemma", "ollama", "custom", "llm"):
      p = self._providers.get(key)
      if p and p.is_ready():
        return p
    return self.provider

  def is_ready(self) -> bool:
    return any(p and p.is_ready() for p in self._providers.values())

  def available_backends(self) -> list[str]:
    return [
      b for b in ("custom", "gemma", "ollama", "llm")
      if self._providers.get(b) and self._providers[b].is_ready()
    ]

  async def chat(
    self,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    backend: str | None = None,
    **kwargs,
  ) -> tuple[str, str]:
    """Returns (reply_text, backend_used)."""
    chosen, cleaned = resolve_backend(
      messages,
      model_field=model,
      default_backend=self._settings.model_backend,
      explicit_backend=backend or kwargs.pop("backend", None),
    )

    last_user = next(
      (m.get("content", "") for m in reversed(cleaned) if m.get("role") == "user"),
      "",
    )
    if is_gold_price_query(last_user):
      live = await fetch_gold_price_context(last_user)
      if live:
        self._last_backend = "live"
        return live, "live"

    if is_title_meta_query(last_user):
      text = await self._title_meta_reply(last_user)
      if text:
        self._last_backend = "title_meta"
        return text, "title_meta"

    if chosen == "auto":
      text, used = await self._chat_auto(cleaned, **kwargs)
    else:
      text, used = await self._chat_one(chosen, cleaned, **kwargs)

    self._last_backend = used
    return text, used

  async def _chat_one(
    self,
    backend: str,
    messages: list[dict[str, str]],
    **kwargs,
  ) -> tuple[str, str]:
    provider = await self._ensure_loaded(backend)
    if not provider or not provider.is_ready():
      raise RuntimeError(
        f"Backend '{backend}' is not available. "
        + _backend_hint(backend)
      )
    return await provider.chat(messages, **kwargs), backend

  async def _title_meta_reply(self, query: str) -> str | None:
    from app.services import title_meta

    provider = await self._ensure_loaded("custom")
    if not provider or not provider.is_ready():
      return None
    topic = extract_title_meta_topic(query)
    try:
      result = await title_meta.generate(
        provider,
        topic=topic,
        variations=3,
        category="blog_article",
        use_ai=True,
      )
      return format_title_meta_reply(result)
    except Exception:
      return None

  async def _chat_auto(self, messages: list[dict[str, str]], **kwargs) -> tuple[str, str]:
    errors: list[str] = []
    order = ["custom", "gemma", "ollama"]
    if self._settings.llm_backend_enabled:
      order.append("llm")
    for backend in order:
      provider = await self._ensure_loaded(backend)
      if not provider or not provider.is_ready():
        errors.append(f"{backend}: not loaded ({_backend_hint(backend)})")
        continue
      try:
        text = await provider.chat(messages, **kwargs)
        if backend == "custom" and is_low_quality_output(text):
          errors.append("custom: low-quality inference (using fallback)")
          continue
        return text, backend
      except Exception as exc:
        errors.append(f"{backend}: {exc}")
        continue
    detail = "; ".join(errors) if errors else "no backends configured"
    raise RuntimeError(f"Auto routing failed — {detail}")

  def model_id(self, backend: str | None = None) -> str:
    key = backend or self._last_backend
    if key == "auto":
      key = self._last_backend
    provider = self._providers.get(key) or self._providers.get("custom")
    if provider and provider.is_ready():
      return provider.model_id()
    return self._settings.model_id


def _backend_hint(backend: str) -> str:
  if backend == "gemma":
    return (
      f"place model.safetensors in {get_settings().gemma_model_dir} "
      f"or run: ollama pull {get_settings().gemma_ollama_model}"
    )
  if backend == "ollama":
    return "install Ollama and run: ollama pull qwen2.5:0.5b"
  if backend == "llm":
    return "run: python scripts/download_model.py"
  return "run: python scripts/train_worldwide.py"


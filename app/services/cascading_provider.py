"""Cascading model provider — try backends in order until one succeeds.

Used by AI tools so they get hosted-LLM quality when available, then
fall back to the custom / local model without failing the request.
"""

from __future__ import annotations

from app.services.provider_base import ModelProvider


class CascadingProvider(ModelProvider):
  def __init__(
    self,
    providers: list[ModelProvider],
    *,
    model_id: str,
    skip_low_quality: bool = True,
  ) -> None:
    self._providers = [p for p in providers if p is not None]
    self._model_id = model_id
    self._skip_low_quality = skip_low_quality
    self._last_backend: str | None = None

  @property
  def last_backend(self) -> str | None:
    return self._last_backend

  async def load(self) -> None:
    return None

  async def unload(self) -> None:
    return None

  def is_ready(self) -> bool:
    return any(p.is_ready() for p in self._providers)

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    if not self._providers:
      raise RuntimeError("No AI providers available for tools")

    from app.services.backend_router import is_low_quality_output

    errors: list[str] = []
    for provider in self._providers:
      if not provider.is_ready():
        continue
      name = type(provider).__name__.replace("Provider", "").lower()
      try:
        text = await provider.chat(messages, **kwargs)
      except Exception as exc:
        errors.append(f"{name}: {exc}")
        continue
      if not (text or "").strip():
        errors.append(f"{name}: empty reply")
        continue
      # Only treat custom-model garbage as a soft failure; hosted replies pass through.
      if (
        self._skip_low_quality
        and "custom" in name
        and is_low_quality_output(text)
      ):
        errors.append(f"{name}: low-quality output")
        continue
      self._last_backend = name
      return text

    detail = "; ".join(errors) if errors else "no ready providers"
    raise RuntimeError(f"All tool AI backends failed — {detail}")

  def model_id(self) -> str:
    return self._model_id

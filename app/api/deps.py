"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.config import get_settings
from app.services.provider_base import ModelProvider
from app.services.registry import ProviderRegistry


def get_registry(request: Request) -> ProviderRegistry:
  return request.app.state.registry


def get_tool_provider(request: Request, model: str | None = None) -> ModelProvider:
  """AI provider for tools — hosted LLM first (like Chat), then custom model."""
  registry = get_registry(request)
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  # Prefer the cascading tool path so SEO/email/resume get ChatGPT-quality polish.
  if model is None or normalize_is_product_model(model):
    return registry.tool_provider()
  return registry.get_provider_for_model(model)


def normalize_is_product_model(model: str) -> bool:
  from app.services.backend_router import normalize_backend

  return normalize_backend(model, "custom") in ("custom", "auto")


def get_model_id(request: Request) -> str:
  settings = get_settings()
  registry = get_registry(request)
  if registry.is_ready():
    try:
      return registry.model_id("custom")
    except RuntimeError:
      pass
  return settings.model_id

"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.config import get_settings
from app.services.provider_base import ModelProvider
from app.services.registry import ProviderRegistry


def get_registry(request: Request) -> ProviderRegistry:
  return request.app.state.registry


def get_tool_provider(request: Request, model: str | None = None) -> ModelProvider:
  """custom-nexus-v1 (custom backend) first; optional model override."""
  registry = get_registry(request)
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.get_provider_for_model(model or get_settings().model_id)


def get_model_id(request: Request) -> str:
  settings = get_settings()
  registry = get_registry(request)
  if registry.is_ready():
    try:
      return registry.model_id("custom")
    except RuntimeError:
      pass
  return settings.model_id

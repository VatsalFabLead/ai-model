"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.services.provider_base import ModelProvider
from app.services.registry import ProviderRegistry


def get_tool_provider(request: Request) -> ModelProvider:
  """Gemma + Nexus RAG for generators; falls back to custom if Gemma unavailable."""
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.tool_provider()

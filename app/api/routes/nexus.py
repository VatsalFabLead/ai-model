"""Unified Nexus API — single entry point for all tools under custom-nexus-v1."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.api.deps import get_model_id, get_registry
from app.config import get_settings
from app.core.security import verify_api_key
from app.services.nexus_gateway import NEXUS_MODEL_ID, NEXUS_TOOL_CATALOG, invoke_nexus_tool

router = APIRouter(prefix="/nexus", tags=["nexus"])


class NexusInvokeRequest(BaseModel):
  model: str = Field(
    default=NEXUS_MODEL_ID,
    description="Model ID — use custom-nexus-v1 for all tools",
    examples=[NEXUS_MODEL_ID],
  )
  tool: str = Field(
    ...,
    description="Tool id: seo_content, chat, email_new, resume_builder, etc.",
    examples=["seo_content", "chat", "email_new"],
  )
  input: dict[str, Any] = Field(default_factory=dict, description="Tool-specific payload")


class NexusInvokeResponse(BaseModel):
  model: str
  tool: str
  result: dict[str, Any]
  elapsed_ms: float


class NexusStatusResponse(BaseModel):
  model_id: str
  model_ready: bool
  tools: list[dict[str, Any]]
  invoke_url: str
  chat_url: str


@router.get("/status", response_model=NexusStatusResponse)
async def nexus_status(
  request: Request,
  _: str = Depends(verify_api_key),
) -> NexusStatusResponse:
  settings = get_settings()
  registry = get_registry(request)
  prefix = settings.api_prefix
  return NexusStatusResponse(
    model_id=get_model_id(request),
    model_ready=registry.is_ready(),
    tools=NEXUS_TOOL_CATALOG,
    invoke_url=f"{prefix}/nexus/invoke",
    chat_url=f"{prefix}/chat/completions",
  )


@router.get("/tools")
async def list_tools(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  return {
    "model": NEXUS_MODEL_ID,
    "tools": NEXUS_TOOL_CATALOG,
    "count": len(NEXUS_TOOL_CATALOG),
  }


@router.post("/invoke", response_model=NexusInvokeResponse)
async def nexus_invoke(
  payload: NexusInvokeRequest,
  request: Request,
  response: Response,
  _: str = Depends(verify_api_key),
) -> NexusInvokeResponse:
  registry = get_registry(request)
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")

  model = (payload.model or NEXUS_MODEL_ID).strip()
  try:
    out = await invoke_nexus_tool(
      registry,
      tool=payload.tool,
      model=model,
      input_data=payload.input,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Tool invocation failed: {exc}") from exc

  response.headers["X-Nexus-Model"] = out["model"]
  response.headers["X-Nexus-Tool"] = out["tool"]
  if payload.tool == "chat" and isinstance(out.get("result"), dict):
    backend = out["result"].get("backend")
    if backend:
      response.headers["X-Nexus-Backend"] = str(backend)

  return NexusInvokeResponse(**out)

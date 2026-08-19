"""Chat page — AI Tools Hub (all generators in one UI)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import get_settings

router = APIRouter(tags=["pages"])

_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "templates" / "chat_hub.html"


@router.get("/chat_page", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
  settings = get_settings()
  html = _TEMPLATE.read_text(encoding="utf-8")
  body = (
    html.replace("{app_name}", str(settings.app_name))
    .replace("{api_prefix}", str(settings.api_prefix))
    .replace("{model_backend}", str(settings.model_backend))
    .replace("{model_id}", str(settings.model_id))
  )
  return HTMLResponse(
    content=body,
    headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
  )

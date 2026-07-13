"""Static HTML API reference — sidebar modules + endpoint detail panels."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.config import get_settings

router = APIRouter(tags=["pages"])

_TEMPLATE = Path(__file__).resolve().parent.parent.parent / "templates" / "api_docs.html"


@router.get("/api-docs", response_class=HTMLResponse)
async def api_docs() -> HTMLResponse:
  settings = get_settings()
  html = _TEMPLATE.read_text(encoding="utf-8")
  html = (
    html.replace("{{APP_NAME}}", settings.app_name)
    .replace("{{API_PREFIX}}", settings.api_prefix)
    .replace("{{LIVE_BASE}}", "https://ai-model-api-2906.onrender.com")
  )
  return HTMLResponse(
    content=html,
    headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
  )

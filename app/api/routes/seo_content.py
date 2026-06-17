"""SEO Content Generator API — create SEO-optimized articles from a topic."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import seo_content
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/seo-content", tags=["seo-content"])


class SeoContentRequest(BaseModel):
  topic: str = Field(..., min_length=1, max_length=300, examples=["Best practices for email marketing"])
  # Accepts a comma-separated string (matches the UI) or a list.
  keywords: list[str] | str | None = Field(default=None, examples=["email marketing, conversions, newsletters"])
  tone: str | None = Field(default=None, examples=["professional", "casual", "informative"])
  word_count: int | None = Field(default=None, ge=100, le=1500)
  audience: str | None = Field(default=None, max_length=200)


class SeoContentResponse(BaseModel):
  title: str
  meta_description: str
  slug: str
  keywords: list[str]
  content: str
  word_count: int


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.post("/generate", response_model=SeoContentResponse)
async def generate(
  payload: SeoContentRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> SeoContentResponse:
  provider = _get_provider(request)
  try:
    result = await seo_content.generate_seo_content(
      provider,
      topic=payload.topic,
      keywords=payload.keywords,
      tone=payload.tone,
      word_count=payload.word_count,
      audience=payload.audience,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO content generation failed: {exc}") from exc
  return SeoContentResponse(**result)

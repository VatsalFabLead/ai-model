"""SEO Content Generator API — single /generate endpoint."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import seo_content
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/seo-content", tags=["seo-content"])

ToneOption = Literal["professional", "casual", "friendly", "formal"]


class SeoContentRequest(BaseModel):
  topic: str = Field(..., min_length=1, max_length=300, examples=["Email marketing best practices"])
  keywords: list[str] | str | None = Field(default=None, examples=["email marketing, conversions, newsletters"])
  tone: ToneOption | None = Field(
    default="professional",
    examples=["professional", "casual", "friendly", "formal"],
    description="Writing tone — only professional, casual, friendly, or formal",
  )
  word_count: int | None = Field(default=500, ge=100, le=1500)
  audience: str | None = Field(default=None, examples=["small business owners", "Surat India"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "how_to_guide", "listicle", "landing_page", "local_seo"],
  )
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(default=True, description="Generate via your custom local model")
  discover_keywords: bool = Field(
    default=False,
    description="Auto-discover related keywords from web search before writing",
  )
  max_keyword_items: int = Field(default=10, ge=3, le=20)


class SeoQuality(BaseModel):
  seo_score: int
  seo_ready: bool
  issues: list[str]
  heading_count: int
  has_conclusion: bool


class SeoDiscoveryMeta(BaseModel):
  enabled: bool
  sources_used: list[str] = Field(default_factory=list)
  keyword_count: int = 0


class SeoAiMeta(BaseModel):
  enabled: bool
  model_used: bool


class SeoContentResponse(BaseModel):
  topic: str
  category: str
  language: str
  tone: str
  title: str
  meta_description: str
  slug: str
  keywords: list[str]
  content: str
  word_count: int
  quality: SeoQuality
  discovery: SeoDiscoveryMeta
  ai: SeoAiMeta


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.get("/categories")
async def list_categories(_: str = Depends(verify_api_key)) -> dict:
  return {"categories": seo_content.supported_categories(), "count": len(seo_content.supported_categories())}


@router.get("/tones")
async def list_tones(_: str = Depends(verify_api_key)) -> dict:
  tones = seo_content.supported_tones()
  return {"tones": tones, "count": len(tones), "allowed": ["professional", "casual", "friendly", "formal"]}


@router.get("/languages")
async def list_languages(_: str = Depends(verify_api_key)) -> dict:
  langs = seo_content.supported_languages()
  return {"languages": langs, "count": len(langs)}


@router.post("/generate", response_model=SeoContentResponse)
async def generate(
  payload: SeoContentRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> SeoContentResponse:
  provider = _get_provider(request)
  try:
    result = await seo_content.generate(
      provider,
      topic=payload.topic,
      keywords=payload.keywords,
      tone=payload.tone,
      word_count=payload.word_count,
      audience=payload.audience,
      category=payload.category,
      language=payload.language,
      use_ai=payload.use_ai,
      discover_keywords=payload.discover_keywords,
      max_keyword_items=payload.max_keyword_items,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO content generation failed: {exc}") from exc
  return SeoContentResponse(**result)

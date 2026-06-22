"""SEO Content Optimizer API — single /optimize endpoint (matches optimizer UI)."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import seo_optimizer
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/seo-optimizer", tags=["seo-optimizer"])

ToneOption = Literal["professional", "casual", "friendly", "formal"]


class ContentMetrics(BaseModel):
  readability_score: float
  word_count: int
  character_count: int
  sentence_count: int


class SeoIssue(BaseModel):
  type: str
  priority: str
  message: str


class OptimizeRequest(BaseModel):
  content: str = Field(..., min_length=1, max_length=12000, description="Paste content to analyze and optimize")
  keywords: list[str] | str | None = Field(default=None, examples=["email marketing, SEO"])
  tone: ToneOption = Field(default="professional", description="professional | casual | friendly | formal")
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "landing_page", "product_description", "email_copy"],
  )
  use_ai: bool = Field(default=True, description="Optimize via your custom local model")


class AiMeta(BaseModel):
  enabled: bool
  model_used: bool


class OptimizeResponse(BaseModel):
  category: str
  language: str
  tone: str
  original: ContentMetrics
  optimized: ContentMetrics
  seo_score_before: int
  seo_score_after: int
  improvement: int
  optimized_content: str
  suggestions: list[str]
  issues_before: list[SeoIssue]
  issues_after: list[SeoIssue]
  keywords: list[str]
  ai: AiMeta


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.get("/categories")
async def list_categories(_: str = Depends(verify_api_key)) -> dict:
  cats = seo_optimizer.supported_categories()
  return {"categories": cats, "count": len(cats)}


@router.get("/tones")
async def list_tones(_: str = Depends(verify_api_key)) -> dict:
  return {"tones": seo_optimizer.supported_tones(), "allowed": ["professional", "casual", "friendly", "formal"]}


@router.get("/languages")
async def list_languages(_: str = Depends(verify_api_key)) -> dict:
  return {"languages": seo_optimizer.supported_languages()}


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_content(
  payload: OptimizeRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> OptimizeResponse:
  provider = _get_provider(request)
  try:
    result = await seo_optimizer.optimize(
      provider,
      content=payload.content,
      keywords=payload.keywords,
      tone=payload.tone,
      language=payload.language,
      category=payload.category,
      use_ai=payload.use_ai,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO optimization failed: {exc}") from exc
  return OptimizeResponse(**result)

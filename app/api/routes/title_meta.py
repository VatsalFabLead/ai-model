"""SEO Title & Meta Description Generator API."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import title_meta
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/title-meta", tags=["title-meta"])

ToneOption = Literal["professional", "casual", "friendly", "formal"]


class TitleMetaVariation(BaseModel):
  title: str
  title_length: int
  meta_description: str
  meta_length: int
  angle: str = ""
  quality_score: int = 0
  seo_ready: bool = False
  issues: list[str] = Field(default_factory=list)


class TitleMetaQuality(BaseModel):
  average_score: int
  seo_ready: bool
  all_ready: bool


class TitleMetaAiMeta(BaseModel):
  enabled: bool
  model_used: bool


class TitleMetaRequest(BaseModel):
  topic: str = Field(..., min_length=1, max_length=300, examples=["Email Marketing Best Practices"])
  variations: int = Field(default=3, ge=1, le=5)
  tone: ToneOption = Field(default="professional", description="professional | casual | friendly | formal")
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "product_page", "landing_page", "local_business", "how_to"],
  )
  use_ai: bool = Field(default=True, description="Generate via your custom local model")


class TitleMetaResponse(BaseModel):
  topic: str
  category: str
  language: str
  tone: str
  variations: list[TitleMetaVariation]
  variation_count: int
  title_limit: int
  meta_min: int
  meta_max: int
  quality: TitleMetaQuality
  ai: TitleMetaAiMeta


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.get("/categories")
async def list_categories(_: str = Depends(verify_api_key)) -> dict:
  cats = title_meta.supported_categories()
  return {"categories": cats, "count": len(cats)}


@router.get("/tones")
async def list_tones(_: str = Depends(verify_api_key)) -> dict:
  return {"tones": title_meta.supported_tones(), "allowed": ["professional", "casual", "friendly", "formal"]}


@router.get("/languages")
async def list_languages(_: str = Depends(verify_api_key)) -> dict:
  return {"languages": title_meta.supported_languages()}


@router.post("/generate", response_model=TitleMetaResponse)
async def generate(
  payload: TitleMetaRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> TitleMetaResponse:
  provider = _get_provider(request)
  try:
    result = await title_meta.generate(
      provider,
      topic=payload.topic,
      variations=payload.variations,
      tone=payload.tone,
      language=payload.language,
      category=payload.category,
      use_ai=payload.use_ai,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Title/meta generation failed: {exc}") from exc
  return TitleMetaResponse(**result)

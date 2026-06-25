"""SEO Title & Meta Description Generator API."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import title_meta

router = APIRouter(prefix="/title-meta", tags=["title-meta"])

ToneOption = Literal["professional", "casual", "friendly", "formal"]


class TitleMetaVariation(BaseModel):
  title: str
  title_length: int
  meta_description: str
  meta_length: int
  angle: str = ""
  quality_score: int = 0
  seo_score: int = 0
  ctr_score: int = 0
  overall_score: int = 0
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
  topic: str = Field(..., min_length=1, max_length=300, examples=["Electric Vehicles in India"])
  variations: int = Field(default=10, ge=10, le=50, description="10–50 unique title+meta pairs per request")
  tone: ToneOption = Field(default="professional", description="professional | casual | friendly | formal")
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "product_page", "landing_page", "local_business", "how_to"],
  )
  use_ai: bool = Field(default=False, description="Optional polish via custom local model")
  use_rag: bool = Field(default=True, description="Use open-dataset SERP/evidence routing")
  variation_seed: int | None = Field(default=None, description="Omit for unique output each request")


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
  generator_version: str | None = None
  variation_seed: int | None = None
  architecture: dict[str, Any] | None = None
  pipeline: dict[str, Any] | None = None
  policy: dict[str, Any] | None = None
  rag: dict[str, Any] | None = None
  elapsed_ms: float | None = None


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


@router.get("/version")
async def title_meta_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.title_meta_rag_pipeline import GENERATOR_VERSION

  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.title_meta_rag_pipeline import ARCHITECTURE_FLOW, OPEN_DATASET_TREE

  return {"flow": ARCHITECTURE_FLOW, "open_datasets": OPEN_DATASET_TREE}


@router.post("/generate", response_model=TitleMetaResponse)
async def generate(
  payload: TitleMetaRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> TitleMetaResponse:
  provider = None
  if payload.use_ai:
    provider = get_tool_provider(request)
  try:
    result = await title_meta.generate(
      provider,
      topic=payload.topic,
      variations=payload.variations,
      tone=payload.tone,
      language=payload.language,
      category=payload.category,
      use_ai=payload.use_ai,
      use_rag=payload.use_rag,
      variation_seed=payload.variation_seed,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Title/meta generation failed: {exc}") from exc
  version = result.get("generator_version", "unknown")
  return JSONResponse(
    content=TitleMetaResponse(**result).model_dump(),
    headers={"X-Title-Meta-Version": version},
  )

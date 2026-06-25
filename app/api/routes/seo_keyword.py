"""SEO Keyword Generator API."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import seo_keyword

router = APIRouter(prefix="/seo-keywords", tags=["seo-keywords"])


class KeywordItem(BaseModel):
  keyword: str
  category: str = "secondary"
  topic_cluster: str = "General"
  volume_estimate: str
  volume_label: str
  volume_range: str
  difficulty_estimate: str
  difficulty_label: str
  cpc_estimate: str
  cpc_label: str
  cpc_range: str
  competition_estimate: str
  competition_label: str
  trend: str
  trend_icon: str = "➜"
  trend_monthly: list[int] = Field(default_factory=list)
  trend_chart: str = ""
  intent: str
  relevance_score: int = 0
  sources: list[str] = Field(default_factory=list)
  seo_score: int = 0
  opportunity_score: int | None = None
  opportunity_breakdown: dict[str, Any] | None = None
  metrics_source: str = "ai_estimate"


class DiscoveryMeta(BaseModel):
  enabled: bool
  sources_used: list[str] = Field(default_factory=list)
  queries_run: int = 0
  errors: list[str] = Field(default_factory=list)


class SeoKeywordRequest(BaseModel):
  seed_keyword: str = Field(
    ...,
    min_length=1,
    examples=["Fablead Developers Technolab"],
    description="Seed keyword, topic, or full brief — no character limit",
  )
  variations: int = Field(default=10, ge=10, le=50, description="10–50 unique keywords per request")
  max_items: int | None = Field(default=None, ge=10, le=50, description="Alias for variations")
  tone: str | None = Field(default=None, examples=["informative", "professional"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(default=False, description="Optional enrichment via local model")
  use_rag: bool = Field(default=True, description="Use open-dataset evidence routing")
  discover_web: bool = Field(default=True, description="Google/Bing suggest, Datamuse, Wikipedia")
  include_questions: bool = Field(default=True)
  include_alphabet: bool = Field(default=True)
  variation_seed: int | None = Field(default=None, description="Omit for unique output each request")


class SeoKeywordResponse(BaseModel):
  seed_keyword: str
  count: int
  summary: dict[str, Any]
  keywords: list[KeywordItem]
  keyword_categories: dict[str, list[KeywordItem]] | None = None
  discovery: DiscoveryMeta
  generator_version: str | None = None
  variation_seed: int | None = None
  metrics_source: str | None = None
  metrics_disclaimer: str | None = None
  architecture: dict[str, Any] | None = None
  pipeline: dict[str, Any] | None = None
  clusters: list[dict[str, Any]] | None = None
  topic_clusters: dict[str, list[Any]] | None = None
  opportunities: list[dict[str, Any]] | None = None
  output: dict[str, Any] | None = None
  recommendations: list[str] | None = None
  seo_score: dict[str, Any] | None = None
  rag: dict[str, Any] | None = None
  elapsed_ms: float | None = None
  ai: dict[str, bool] | None = None


@router.get("/version")
async def keyword_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.seo_keyword_rag_pipeline import GENERATOR_VERSION

  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.seo_keyword_enrichment import ARCHITECTURE_FLOW, OPEN_DATASET_TREE
  from app.engine.seo_keyword_domains import DOMAIN_CATALOG, DOMAIN_COUNT, MASTER_DOMAINS
  from app.engine.seo_keyword_open_data import DATASET_STACK
  from app.engine.seo_keyword_rag_pipeline import GENERATOR_VERSION

  return {
    "version": GENERATOR_VERSION,
    "flow": ARCHITECTURE_FLOW,
    "open_datasets": OPEN_DATASET_TREE,
    "dataset_stack": DATASET_STACK,
    "domain_catalog": DOMAIN_CATALOG,
    "domain_count": DOMAIN_COUNT,
    "stages": len(ARCHITECTURE_FLOW),
  }


@router.post("/generate", response_model=SeoKeywordResponse)
async def generate(
  payload: SeoKeywordRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> SeoKeywordResponse:
  provider = None
  if payload.use_ai:
    provider = get_tool_provider(request)
  variations = payload.variations if payload.max_items is None else payload.max_items
  try:
    result = await seo_keyword.generate_keywords(
      provider,
      seed_keyword=payload.seed_keyword,
      tone=payload.tone,
      variations=variations,
      language=payload.language,
      use_ai=payload.use_ai,
      use_rag=payload.use_rag,
      discover_web=payload.discover_web,
      include_questions=payload.include_questions,
      include_alphabet=payload.include_alphabet,
      variation_seed=payload.variation_seed,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO keyword generation failed: {exc}") from exc
  version = result.get("generator_version", "unknown")
  return JSONResponse(
    content=SeoKeywordResponse(**result).model_dump(),
    headers={"X-SEO-Keyword-Version": version},
  )

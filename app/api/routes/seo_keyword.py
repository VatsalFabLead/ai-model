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
  semantic_similarity: int | None = None
  confidence_score: int | None = None
  reason: str | None = None


class DiscoveryMeta(BaseModel):
  enabled: bool
  sources_used: list[str] = Field(default_factory=list)
  queries_run: int = 0
  errors: list[str] = Field(default_factory=list)


class SeoKeywordRequest(BaseModel):
  seed_keyword: str | None = Field(
    default=None,
    examples=["Navio Coffee"],
    description="Primary seed / topic (optional if title/content/context provided)",
  )
  title: str | None = Field(default=None, description="Article or page title")
  content: str | None = Field(
    default=None,
    description="Full article/page content — preferred input for strategist mode",
  )
  context: str | None = Field(
    default=None,
    examples=[
      "We run a specialty coffee brand in Surat selling single-origin beans, "
      "subscriptions, and cafe brewing workshops for beginners."
    ],
    description="Business / topic brief (alias for content when content is empty)",
  )
  primary_topic: str | None = Field(default=None, description="Optional primary topic override")
  country: str | None = Field(default=None, examples=["India", "United States"], description="Market / country")
  market: str | None = Field(default=None, description="Alias for country")
  variations: int = Field(default=10, ge=10, le=50, description="10–50 unique keywords in flat list")
  max_items: int | None = Field(default=None, ge=10, le=50, description="Alias for variations")
  tone: str | None = Field(default=None, examples=["informative", "professional"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(
    default=True,
    description="Use hosted SEO strategist (Groq) when content/title provided; else pipeline",
  )
  mode: str | None = Field(
    default=None,
    description="strategist | pipeline — default auto (strategist when AI+content)",
  )
  use_rag: bool = Field(default=True, description="Use open-dataset evidence routing (pipeline mode)")
  discover_web: bool = Field(default=True, description="Google/Bing suggest, Datamuse, Wikipedia (pipeline mode)")
  include_questions: bool = Field(default=True)
  include_alphabet: bool = Field(default=True)
  variation_seed: int | None = Field(default=None, description="Omit for unique output each request")


class SeoKeywordResponse(BaseModel):
  seed_keyword: str
  context: str | None = None
  title: str | None = None
  primary_topic: str | None = None
  market: str | None = None
  language: str | None = None
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
  ai: dict[str, Any] | None = None
  strategist: dict[str, Any] | None = None


@router.get("/version")
async def keyword_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.seo_keyword_rag_pipeline import GENERATOR_VERSION
  from app.engine.seo_keyword_strategist import STRATEGIST_VERSION

  return {
    "generator_version": GENERATOR_VERSION,
    "strategist_version": STRATEGIST_VERSION,
    "status": "ok",
  }


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
      seed_keyword=payload.seed_keyword or "",
      context=payload.context,
      title=payload.title,
      content=payload.content,
      primary_topic=payload.primary_topic,
      country=payload.country,
      market=payload.market,
      tone=payload.tone,
      variations=variations,
      language=payload.language,
      use_ai=payload.use_ai,
      mode=payload.mode,
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

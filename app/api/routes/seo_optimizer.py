"""SEO Content Optimizer API — RAG pipeline + /optimize endpoint."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import seo_optimizer

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
  keywords: list[str] | str | None = Field(
    default=None,
    description="Target keywords — comma, newline, semicolon, or pipe separated; any phrasing accepted",
    examples=["ERP software, manufacturing ERP", "flutter app development"],
  )
  tone: ToneOption = Field(default="professional", description="professional | casual | friendly | formal")
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "landing_page", "product_description", "email_copy"],
  )
  use_ai: bool = Field(default=True, description="Polish RAG output with your custom local model")
  use_rag: bool = Field(default=True, description="Use open-dataset RAG pipeline (Wikipedia, Wikidata, etc.)")
  variation_seed: int | None = Field(default=None, description="Omit for unique output each request")


class AiMeta(BaseModel):
  enabled: bool
  model_used: bool


class RagMeta(BaseModel):
  enabled: bool
  sources_used: list[str] = Field(default_factory=list)
  confidence: float = 0.0


class GapItem(BaseModel):
  type: str
  source: str
  suggestion: str


class InternalLink(BaseModel):
  anchor_text: str
  target_topic: str
  reason: str


class FaqItem(BaseModel):
  question: str
  answer: str


class MetadataOpt(BaseModel):
  title: str
  meta_description: str


class PipelineAnalysis(BaseModel):
  keyword_analysis: dict[str, Any] = Field(default_factory=dict)
  entity_extraction: list[str] | dict[str, Any] = Field(default_factory=list)
  coverage_map: dict[str, Any] = Field(default_factory=dict)
  gap_analysis: list[dict[str, Any]] = Field(default_factory=list)
  source_router: dict[str, Any] = Field(default_factory=dict)
  retrieval: dict[str, Any] = Field(default_factory=dict)
  novelty: dict[str, Any] = Field(default_factory=dict)
  section_plan: list[dict[str, Any]] = Field(default_factory=list)
  readability_analysis: dict[str, Any] = Field(default_factory=dict)


class OptimizationBundle(BaseModel):
  metadata: MetadataOpt
  internal_links: list[InternalLink] = Field(default_factory=list)
  faqs: list[FaqItem] = Field(default_factory=list)
  schema_suggestions: dict[str, Any] = Field(default_factory=dict)


class MetricsComparison(BaseModel):
  original: ContentMetrics
  optimized: ContentMetrics


class OptimizeResponse(BaseModel):
  category: str
  language: str
  tone: str
  original: ContentMetrics
  optimized: ContentMetrics
  metrics: MetricsComparison | None = None
  seo_score_before: int
  seo_score_after: int
  improvement: int
  optimized_content: str
  suggestions: list[str]
  issues_before: list[SeoIssue]
  issues_after: list[SeoIssue]
  keywords: list[str]
  ai: AiMeta
  use_rag: bool = True
  generator_version: str = "seo-optimizer-rag-v5.2"
  variation_seed: int | None = None
  rag: RagMeta | None = None
  pipeline: PipelineAnalysis | None = None
  architecture: dict[str, Any] | None = None
  optimization: OptimizationBundle | None = None


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


@router.get("/version")
async def optimizer_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.seo_optimizer_rag_pipeline import GENERATOR_VERSION

  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.seo_optimizer_rag_pipeline import ARCHITECTURE_FLOW

  return {
    "flow": ARCHITECTURE_FLOW,
    "datasets": [
      "Wikipedia", "Wikidata", "DBpedia", "ConceptNet", "Stack Exchange",
      "arXiv", "Semantic Scholar", "GDELT", "GooAQ", "SQuAD", "Dolly", "C4", "FineWeb",
    ],
    "routes": {
      "general": ["wikipedia", "wikidata", "dbpedia"],
      "technical": ["arxiv", "semantic_scholar", "stackexchange", "wikipedia"],
      "news": ["gdelt", "fineweb", "wikipedia"],
      "enterprise": ["wikipedia", "wikidata", "stackexchange", "semantic_scholar"],
    },
  }


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize_content(
  payload: OptimizeRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> OptimizeResponse:
  provider = None
  if payload.use_ai:
    provider = get_tool_provider(request)
  try:
    result = await seo_optimizer.optimize(
      provider,
      content=payload.content,
      keywords=payload.keywords,
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
    raise HTTPException(status_code=500, detail=f"SEO optimization failed: {exc}") from exc
  version = result.get("generator_version", "unknown")
  return JSONResponse(
    content=OptimizeResponse(**result).model_dump(),
    headers={"X-SEO-Optimizer-Version": version},
  )

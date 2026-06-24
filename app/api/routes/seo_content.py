"""SEO Content Generator API — single /generate endpoint."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import seo_content

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
  word_count: int | None = Field(default=1000, ge=100, le=2500)
  audience: str | None = Field(default=None, examples=["small business owners", "Surat India"])
  category: str | None = Field(
    default="blog_article",
    examples=["blog_article", "how_to_guide", "listicle", "landing_page", "local_seo"],
  )
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(default=True, description="Enhance template with your custom local model")
  discover_keywords: bool = Field(
    default=False,
    description="Auto-discover related keywords from web search before writing",
  )
  max_keyword_items: int = Field(default=10, ge=3, le=20)
  variation_seed: int | None = Field(
    default=None,
    description="Optional seed for content variation; omit for unique output each request",
  )
  use_rag: bool = Field(
    default=True,
    description="Retrieve from free open datasets (Wikipedia, Wikidata, arXiv, etc.) before generating",
  )


class SeoRagMeta(BaseModel):
  enabled: bool
  topic_class: str | None = None
  confidence: float = 0.0
  sources_routed: list[str] = Field(default_factory=list)
  sources_used: list[str] = Field(default_factory=list)
  document_count: int = 0
  fact_count: int = 0
  entities: list[str] = Field(default_factory=list)
  variation_seed: int | None = None


class SeoKeywords(BaseModel):
  primary: str
  secondary: list[str] = Field(default_factory=list)


class SeoOutlineItem(BaseModel):
  level: str = Field(description="h1, h2, or h3")
  text: str


class SeoMetadata(BaseModel):
  title: str
  meta_description: str


class SeoContentBody(BaseModel):
  article: str
  tone: str


class SeoFaq(BaseModel):
  question: str
  answer: str


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
  metadata: SeoMetadata
  keywords: SeoKeywords
  keywords_list: list[str] = Field(default_factory=list)
  outline: list[SeoOutlineItem]
  outline_text: list[str] = Field(default_factory=list)
  content: SeoContentBody
  article: str
  faqs: list[SeoFaq]
  tone: str
  title: str
  meta_description: str
  slug: str
  word_count: int
  quality: SeoQuality
  discovery: SeoDiscoveryMeta
  ai: SeoAiMeta
  rag: SeoRagMeta | None = None
  generator_version: str = "seo-content-rag-v4.1"
  variation_seed: int | None = None
  domain: str | None = None


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
  provider = None
  if payload.use_ai:
    provider = get_tool_provider(request)
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
      variation_seed=payload.variation_seed,
      use_rag=payload.use_rag,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO content generation failed: {exc}") from exc
  return SeoContentResponse(**result)


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  """SEO content workflow: Keyword → Intent → Entity → Facts → Outline → Writing → SEO → Final."""
  from app.engine.seo_content_rag_pipeline import (
    ARCHITECTURE_FLOW,
    GENERATOR_VERSION,
    WORKFLOW_LABELS,
  )
  return {
    "version": GENERATOR_VERSION,
    "flow": ARCHITECTURE_FLOW,
    "labels": WORKFLOW_LABELS,
    "workflow": [
      {"stage": stage, "label": WORKFLOW_LABELS[stage]}
      for stage in ARCHITECTURE_FLOW
    ],
    "datasets": [
      "Wikipedia", "Wikidata", "DBpedia", "ConceptNet", "Stack Exchange",
      "arXiv", "Semantic Scholar", "GDELT", "GooAQ", "SQuAD", "local FAISS",
    ],
  }


@router.get("/version")
async def generator_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.seo_content_rag_pipeline import GENERATOR_VERSION
  return {"generator_version": GENERATOR_VERSION}


@router.get("/schema")
async def output_schema(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  """Document the structured output tree."""
  return {
    "structure": {
      "metadata": {"title": "string", "meta_description": "string"},
      "keywords": {"primary": "string", "secondary": ["string"]},
      "outline": [{"level": "h1|h2|h3", "text": "string"}],
      "content": {"article": "markdown string", "tone": "string"},
      "faqs": [{"question": "string", "answer": "string"}],
    },
    "speed": {
      "use_ai_false": "Template-only — instant, no model load",
      "use_ai_true": "Template-first + optional custom model polish (timeout 14s)",
    },
  }

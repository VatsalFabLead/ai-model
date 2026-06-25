"""Schema Markup Generator API — advanced, multilingual, worldwide."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import schema_markup

router = APIRouter(prefix="/schema-markup", tags=["schema-markup"])


class SchemaMarkupRequest(BaseModel):
  schema_type: str = Field(
    ...,
    examples=["Article", "Product", "FAQPage", "LocalBusiness", "Recipe", "JobPosting"],
    description=(
      "Schema.org type — Article, Product, LocalBusiness, FAQPage, Organisation, Recipe, "
      "Person, Website, WebPage, Blog, NewsArticle, HowTo, Review, AggregateRating, "
      "Service, MedicalBusiness, Restaurant, RealEstateAgent, Course, JobPosting, "
      "VideoObject, ImageObject, PodcastEpisode, Offer, Brand, BreadcrumbList, SitelinksSearchBox"
    ),
  )
  name: str = Field(..., min_length=1, max_length=300, examples=["Food Delivery App Development Guide"])
  data: dict[str, Any] = Field(default_factory=dict)
  language: str | None = Field(
    default=None,
    examples=["English", "Hindi", "Spanish", "French", "Arabic", "Japanese"],
    description="Page language — sets inLanguage and localized labels",
  )
  ai_enhance: bool = Field(
    default=False,
    description="Enhance via your local custom model + schema training knowledge",
  )
  use_rag: bool = Field(
    default=False,
    description="Enrich from open datasets (Wikipedia, Wikidata, GooAQ) — adds ~6s when enabled",
  )


class SchemaQuality(BaseModel):
  completeness_score: int
  schema_score: int = 0
  google_compliance_score: int = 0
  seo_score: int = 0
  overall_score: int = 0
  seo_ready: bool
  missing_recommended_fields: list[str]
  field_count: int


class SchemaMarkupResponse(BaseModel):
  schema_type: str
  category: str
  language: str
  jsonld: dict[str, Any]
  jsonld_string: str
  quality: SchemaQuality
  generator_version: str | None = None
  validation: dict[str, Any] | None = None
  entities: list[str] | None = None
  requirements: dict[str, Any] | None = None
  seo_analysis: dict[str, Any] | None = None
  architecture: dict[str, Any] | None = None
  elapsed_ms: float | None = None


@router.get("/types")
async def list_types(
  category: str | None = None,
  _: str = Depends(verify_api_key),
) -> dict:
  types = schema_markup.supported_types()
  if category:
    cat = category.strip().lower()
    types = [t for t in types if t.get("category") == cat]
  return {"types": types, "count": len(types)}


@router.get("/categories")
async def list_categories(_: str = Depends(verify_api_key)) -> dict:
  cats = schema_markup.supported_categories()
  return {"categories": cats, "count": len(cats)}


@router.get("/languages")
async def list_languages(_: str = Depends(verify_api_key)) -> dict:
  langs = schema_markup.supported_languages()
  return {"languages": langs, "count": len(langs)}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.schema_markup_rag_pipeline import (
    ARCHITECTURE_FLOW,
    GENERATOR_VERSION,
    OPEN_STANDARDS,
  )
  from app.engine.schema_markup_enrichment import OPEN_DATASET_TREE
  return {
    "version": GENERATOR_VERSION,
    "flow": ARCHITECTURE_FLOW,
    "open_standards": OPEN_STANDARDS,
    "open_datasets": OPEN_DATASET_TREE,
    "validation_layers": [
      "input_validation",
      "schema_validation",
      "seo_validation",
    ],
  }


@router.get("/properties")
async def property_spec(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.schema_markup_enrichment import OPTIONAL_PROPERTIES, REQUIRED_PROPERTIES

  types = sorted(set(REQUIRED_PROPERTIES) | set(OPTIONAL_PROPERTIES))
  spec = []
  for stype in types:
    spec.append({
      "schema_type": stype,
      "required_properties": REQUIRED_PROPERTIES.get(stype, ["name"]),
      "recommended_properties": OPTIONAL_PROPERTIES.get(stype, []),
    })
  return {"types": spec, "count": len(spec)}


@router.get("/version")
async def generator_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.schema_markup_rag_pipeline import GENERATOR_VERSION
  return {"generator_version": GENERATOR_VERSION}


@router.post("/generate", response_model=SchemaMarkupResponse)
async def generate_schema(
  payload: SchemaMarkupRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> SchemaMarkupResponse:
  provider = get_tool_provider(request)
  try:
    result = await schema_markup.generate_schema_markup(
      provider,
      schema_type=payload.schema_type,
      name=payload.name,
      data=payload.data,
      language=payload.language,
      ai_enhance=payload.ai_enhance,
      use_rag=payload.use_rag,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Schema markup generation failed: {exc}") from exc
  return SchemaMarkupResponse(**result)

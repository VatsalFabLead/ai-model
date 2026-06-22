"""SEO Keyword Generator API."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import seo_keyword
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/seo-keywords", tags=["seo-keywords"])


class KeywordItem(BaseModel):
  keyword: str
  search_volume: int
  difficulty: int
  cpc_usd: float
  competition: int
  trend: str
  intent: str
  relevance_score: int = Field(description="0-100 match strength vs seed keyword")
  sources: list[str] = Field(default_factory=list, description="Where this keyword was discovered")


class DiscoveryMeta(BaseModel):
  enabled: bool
  sources_used: list[str] = Field(default_factory=list)
  queries_run: int = 0
  errors: list[str] = Field(default_factory=list)


class SeoKeywordRequest(BaseModel):
  seed_keyword: str = Field(..., min_length=1, max_length=200, examples=["digital marketing"])
  tone: str | None = Field(default=None, examples=["informative", "professional"])
  max_items: int = Field(default=20, ge=5, le=50)
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(default=True, description="Enrich results with the local model")
  discover_web: bool = Field(
    default=True,
    description="Search Google/Bing suggest, Datamuse, and Wikipedia for real related keywords",
  )
  include_questions: bool = Field(
    default=True,
    description='Expand with question patterns like "how to …", "what is …"',
  )
  include_alphabet: bool = Field(
    default=True,
    description="Google alphabet soup (seed + a-z) for long-tail discovery",
  )


class SeoKeywordResponse(BaseModel):
  seed_keyword: str
  count: int
  summary: dict[str, int]
  keywords: list[KeywordItem]
  discovery: DiscoveryMeta


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.post("/generate", response_model=SeoKeywordResponse)
async def generate(
  payload: SeoKeywordRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> SeoKeywordResponse:
  provider = _get_provider(request)
  try:
    result = await seo_keyword.generate_keywords(
      provider,
      seed_keyword=payload.seed_keyword,
      tone=payload.tone,
      max_items=payload.max_items,
      language=payload.language,
      use_ai=payload.use_ai,
      discover_web=payload.discover_web,
      include_questions=payload.include_questions,
      include_alphabet=payload.include_alphabet,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"SEO keyword generation failed: {exc}") from exc
  return SeoKeywordResponse(**result)

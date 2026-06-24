"""Professional Cover Letter AI Generator API."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import cover_letter
from app.services.cover_letter import _ensure_output_fields

router = APIRouter(prefix="/cover-letter", tags=["cover-letter"])


class CoverLetterRequest(BaseModel):
  job_role: str = Field(..., min_length=1, max_length=200, examples=["Senior Software Engineer"])
  company_name: str = Field(..., min_length=1, max_length=200, examples=["Microsoft"])
  skills_experience: str = Field(
    ...,
    min_length=1,
    max_length=6000,
    examples=["5 years Flutter development, led mobile team, shipped 3 apps to production..."],
  )
  tone: str | None = Field(default=None, examples=["professional", "enthusiastic", "confident"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  applicant_name: str | None = Field(default=None, max_length=120)
  use_ai: bool = Field(default=True, description="LLM rewrite for introduction, experience, and skills")
  use_rag: bool = Field(default=True, description="Use open-dataset company/role routing")
  variation_seed: int | None = Field(default=None, description="Omit for unique wording each request")


class CoverLetterQuality(BaseModel):
  quality_score: int
  readability_score: int
  word_count: int
  ats_coverage_pct: int
  letter_ready: bool
  checks_passed: list[str] = Field(default_factory=list)


class CoverLetterAiMeta(BaseModel):
  enabled: bool
  model_used: bool = False
  full_letter_llm: bool = False
  fallback_rule_based: bool = False


class CoverLetterExportMeta(BaseModel):
  pdf_available: bool = False
  docx_available: bool = False
  pdf_base64: str | None = None
  docx_base64: str | None = None


class CoverLetterResponse(BaseModel):
  job_role: str
  company_name: str
  tone: str
  cover_letter: str
  cover_letter_markdown: str
  word_count: int
  skills_list: list[str] = Field(default_factory=list)
  ats_keywords: dict[str, Any] | None = None
  quality: CoverLetterQuality
  ai: CoverLetterAiMeta
  generator_version: str | None = None
  variation_seed: int | None = None
  architecture: dict[str, Any] | None = None
  pipeline: dict[str, Any] | None = None
  export: CoverLetterExportMeta | None = None
  elapsed_ms: float | None = None


@router.get("/version")
async def cover_letter_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.cover_letter_rag_pipeline import GENERATOR_VERSION

  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.cover_letter_rag_pipeline import ARCHITECTURE_FLOW, OPEN_DATASET_TREE
  from app.engine.cover_letter_templates import template_combination_counts

  return {
    "flow": ARCHITECTURE_FLOW,
    "open_datasets": OPEN_DATASET_TREE,
    "template_combinations": template_combination_counts(),
  }


@router.post("/generate", response_model=CoverLetterResponse)
async def generate(
  payload: CoverLetterRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> CoverLetterResponse:
  provider = None
  if payload.use_ai:
    registry = request.app.state.registry
    for key in ("custom", "gemma", "ollama", "llm"):
      p = registry._providers.get(key)
      if p and p.is_ready():
        provider = p
        break
    if provider is None:
      try:
        provider = get_tool_provider(request)
      except HTTPException:
        pass
  try:
    result = await cover_letter.generate_cover_letter(
      provider,
      job_role=payload.job_role,
      company_name=payload.company_name,
      skills_experience=payload.skills_experience,
      tone=payload.tone,
      language=payload.language,
      applicant_name=payload.applicant_name,
      use_ai=payload.use_ai,
      use_rag=payload.use_rag,
      variation_seed=payload.variation_seed,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Cover letter generation failed: {exc}") from exc

  result = _ensure_output_fields(result)
  version = result.get("generator_version", "unknown")
  return JSONResponse(
    content=CoverLetterResponse(**result).model_dump(),
    headers={"X-Cover-Letter-Version": version},
  )

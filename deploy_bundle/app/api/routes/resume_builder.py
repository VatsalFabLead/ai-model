"""Resume Builder API — RAG pipeline with AI + ATS output."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import resume_builder

router = APIRouter(prefix="/resume-builder", tags=["resume-builder"])


class ResumeGenerateRequest(BaseModel):
  full_name: str = Field(..., min_length=1, max_length=120, examples=["Vatsal Patel"])
  job_title: str = Field(..., min_length=1, max_length=120, examples=["Flutter Developer"])
  email: str = Field(..., min_length=3, max_length=200, examples=["you@example.com"])
  phone: str = Field(..., min_length=3, max_length=40, examples=["+91-9876543210"])
  linkedin: str | None = Field(default=None, examples=["https://linkedin.com/in/yourname"])
  portfolio: str | None = Field(default=None, examples=["https://github.com/yourname"])
  education: str | None = Field(default=None, examples=["B.Tech CS — GTU, 2022"])
  experience: str | None = Field(default=None, examples=["Built Flutter apps, integrated Firebase"])
  skills: str | None = Field(default=None, examples=["Flutter, Dart, Firebase, Git"])
  summary: str | None = Field(default=None, description="Leave empty for AI-generated summary")
  projects: str | None = None
  certifications: str | None = None
  achievements: str | None = None
  languages: str | None = Field(default=None, examples=["English (Fluent), Hindi (Native)"])
  template: str = Field(default="modern", examples=["modern", "classic", "executive", "minimal", "creative"])
  template_name: str | None = Field(default=None, description="Alias for template")
  improve: bool = Field(default=False, description="Re-enhance all sections even when filled")
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(default=False, description="Optional polish via local Nexus/Gemma model")
  use_rag: bool = Field(default=True, description="Use open-dataset occupation/skill routing")
  variation_seed: int | None = Field(default=None, description="Omit for unique wording each request")


class ResumeQuality(BaseModel):
  completeness_score: int
  resume_ready: bool
  filled_fields: list[str] = Field(default_factory=list)
  missing_fields: list[str] = Field(default_factory=list)
  field_count: int = 0
  ats_score: int | None = None


class ResumeAiMeta(BaseModel):
  enabled: bool
  skills_generated: bool
  summary_generated: bool
  experience_enhanced: bool
  model_used: bool = False


class ResumeExportMeta(BaseModel):
  pdf_available: bool = False
  docx_available: bool = False
  pdf_base64: str | None = None
  docx_base64: str | None = None


class ResumeGenerateResponse(BaseModel):
  full_name: str
  job_title: str
  email: str
  phone: str
  linkedin: str
  portfolio: str
  education: str
  experience: str
  skills: str
  summary: str
  projects: str
  certifications: str
  achievements: str
  languages: str
  skills_list: list[str]
  experience_bullets: list[str]
  template: str
  template_name: str
  language: str
  category: str
  resume_markdown: str
  resume_ai_text: str
  ats_resume_text: str
  word_count: int
  quality: ResumeQuality
  ai: ResumeAiMeta
  generator_version: str | None = None
  variation_seed: int | None = None
  architecture: dict[str, Any] | None = None
  pipeline: dict[str, Any] | None = None
  ats_keywords: dict[str, Any] | None = None
  skill_groups: dict[str, list[str]] | None = None
  export: ResumeExportMeta | None = None
  elapsed_ms: float | None = None


@router.get("/templates")
async def list_templates(_: str = Depends(verify_api_key)) -> dict:
  return {"templates": resume_builder.supported_templates()}


@router.get("/version")
async def resume_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  from app.engine.resume_rag_pipeline import GENERATOR_VERSION

  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  from app.engine.resume_rag_pipeline import ARCHITECTURE_FLOW, OPEN_DATASET_TREE

  return {"flow": ARCHITECTURE_FLOW, "open_datasets": OPEN_DATASET_TREE}


@router.post("/generate", response_model=ResumeGenerateResponse)
async def generate(
  payload: ResumeGenerateRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> ResumeGenerateResponse:
  provider = None
  if payload.use_ai:
    provider = get_tool_provider(request)
  try:
    result = await resume_builder.generate(
      provider,
      full_name=payload.full_name,
      job_title=payload.job_title,
      email=payload.email,
      phone=payload.phone,
      linkedin=payload.linkedin,
      portfolio=payload.portfolio,
      education=payload.education,
      experience=payload.experience,
      skills=payload.skills,
      summary=payload.summary,
      projects=payload.projects,
      certifications=payload.certifications,
      achievements=payload.achievements,
      languages=payload.languages,
      template=payload.template,
      template_name=payload.template_name,
      language=payload.language,
      use_ai=payload.use_ai,
      improve=payload.improve,
      use_rag=payload.use_rag,
      variation_seed=payload.variation_seed,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Resume generation failed: {exc}") from exc

  result["template_name"] = result.get("template_name") or result.get("template", "modern")
  version = result.get("generator_version", "unknown")
  return JSONResponse(
    content=ResumeGenerateResponse(**result).model_dump(),
    headers={"X-Resume-Builder-Version": version},
  )

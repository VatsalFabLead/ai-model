"""Resume Builder API — single /generate endpoint (like seo-keywords/generate)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import resume_builder

router = APIRouter(prefix="/resume-builder", tags=["resume-builder"])


class ResumeGenerateRequest(BaseModel):
  """All resume fields in one flat request — AI fills missing skills/summary/experience."""

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
  improve: bool = Field(
    default=False,
    description="Re-enhance all sections (summary, experience, skills) even when already filled",
  )
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])
  use_ai: bool = Field(
    default=True,
    description="Auto-generate skills, summary, and optimize experience via custom model",
  )


class ResumeQuality(BaseModel):
  completeness_score: int
  resume_ready: bool
  filled_fields: list[str]
  missing_fields: list[str]
  field_count: int


class ResumeAiMeta(BaseModel):
  enabled: bool
  skills_generated: bool
  summary_generated: bool
  experience_enhanced: bool


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
  language: str
  category: str
  resume_markdown: str
  word_count: int
  quality: ResumeQuality
  ai: ResumeAiMeta


@router.post("/generate", response_model=ResumeGenerateResponse)
async def generate(
  payload: ResumeGenerateRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> ResumeGenerateResponse:
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
      language=payload.language,
      use_ai=payload.use_ai,
      improve=payload.improve,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Resume generation failed: {exc}") from exc

  fields = result.pop("fields", {})
  return ResumeGenerateResponse(
    full_name=fields.get("full_name", payload.full_name),
    job_title=fields.get("job_title", payload.job_title),
    email=fields.get("email", payload.email),
    phone=fields.get("phone", payload.phone),
    linkedin=fields.get("linkedin", "") or "",
    portfolio=fields.get("portfolio", "") or "",
    education=fields.get("education", "") or "",
    experience=fields.get("experience", "") or "",
    skills=fields.get("skills", "") or "",
    summary=result.get("summary", "") or "",
    projects=fields.get("projects", "") or "",
    certifications=fields.get("certifications", "") or "",
    achievements=fields.get("achievements", "") or "",
    languages=fields.get("languages", "") or "",
    skills_list=result.get("skills_list", []),
    experience_bullets=result.get("experience_bullets", []),
    template=result["template"],
    language=result["language"],
    category=result["category"],
    resume_markdown=result["resume_markdown"],
    word_count=result["word_count"],
    quality=result["quality"],
    ai=result["ai"],
  )

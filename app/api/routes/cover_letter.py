"""Professional Cover Letter Builder API."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import cover_letter
from app.services.registry import ProviderRegistry

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


class CoverLetterResponse(BaseModel):
  job_role: str
  company_name: str
  tone: str
  cover_letter: str
  word_count: int


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.post("/generate", response_model=CoverLetterResponse)
async def generate(
  payload: CoverLetterRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> CoverLetterResponse:
  provider = _get_provider(request)
  try:
    result = await cover_letter.generate_cover_letter(
      provider,
      job_role=payload.job_role,
      company_name=payload.company_name,
      skills_experience=payload.skills_experience,
      tone=payload.tone,
      language=payload.language,
      applicant_name=payload.applicant_name,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Cover letter generation failed: {exc}") from exc
  return CoverLetterResponse(**result)

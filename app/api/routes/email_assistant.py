"""AI Email Assistant API — New Email, Reply, Cold Email."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.engine.email_assistant_enrichment import ARCHITECTURE_FLOW, GENERATOR_VERSION
from app.services import email_assistant

router = APIRouter(prefix="/email-assistant", tags=["email-assistant"])

_EMAIL_TONES = ("professional", "casual", "friendly", "formal")


class EmailScores(BaseModel):
  grammar: int = 0
  readability: int = 0
  spam_score: int = 0
  spam_risk: str = "low"
  professionalism: int = 0
  overall: int = 0


class EmailQuality(BaseModel):
  overall: int = 0
  grammar: int = 0
  readability: int = 0
  spam: int = 0
  professionalism: int = 0


class EmailResponse(BaseModel):
  mode: str
  subject: str
  tone: str
  email: str
  word_count: int
  generator_version: str | None = None
  subject_options: list[str] = Field(default_factory=list)
  quality: EmailQuality | dict[str, Any] | None = None
  scores: EmailScores | dict[str, Any] | None = None
  suggestions: list[str] = Field(default_factory=list)
  alternatives: list[dict[str, str]] = Field(default_factory=list)
  architecture: dict[str, Any] | None = None
  pipeline: dict[str, Any] | None = None
  ai: dict[str, bool] | None = None
  elapsed_ms: float | None = None


class NewEmailRequest(BaseModel):
  subject: str = Field(default="", max_length=200)
  context: str = Field(..., min_length=1, max_length=3000, description="Context / key points")
  tone: str | None = Field(default="professional", examples=list(_EMAIL_TONES))


class ReplyEmailRequest(BaseModel):
  original_email: str = Field(..., min_length=1, max_length=6000, description="Original email to reply to")
  reply_points: str = Field(default="", max_length=3000, description="Key points for the reply")
  tone: str | None = Field(default="professional", examples=list(_EMAIL_TONES))


class ColdEmailRequest(BaseModel):
  company_name: str = Field(..., min_length=1, max_length=200)
  purpose_offer: str = Field(..., min_length=1, max_length=1000, description="Purpose / offer")
  value_proposition: str = Field(..., min_length=1, max_length=2000)
  tone: str | None = Field(default="professional", examples=list(_EMAIL_TONES))


@router.get("/version")
async def email_version(_: str = Depends(verify_api_key)) -> dict[str, str]:
  return {"generator_version": GENERATOR_VERSION, "status": "ok"}


@router.get("/pipeline")
async def pipeline_architecture(_: str = Depends(verify_api_key)) -> dict[str, Any]:
  return {
    "version": GENERATOR_VERSION,
    "flow": ARCHITECTURE_FLOW,
    "stages": len(ARCHITECTURE_FLOW),
    "modes": ["new_email", "reply", "cold_email"],
    "tones": list(_EMAIL_TONES),
  }


@router.post("/new-email", response_model=EmailResponse)
async def new_email(
  payload: NewEmailRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> EmailResponse:
  provider = get_tool_provider(request)
  try:
    result = await email_assistant.generate_new_email(
      provider,
      subject=payload.subject,
      context=payload.context,
      tone=payload.tone,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"New email generation failed: {exc}") from exc
  return EmailResponse(**result)


@router.post("/reply", response_model=EmailResponse)
async def reply_email(
  payload: ReplyEmailRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> EmailResponse:
  provider = get_tool_provider(request)
  try:
    result = await email_assistant.generate_reply_email(
      provider,
      original_email=payload.original_email,
      reply_points=payload.reply_points,
      tone=payload.tone,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Reply generation failed: {exc}") from exc
  return EmailResponse(**result)


@router.post("/cold-email", response_model=EmailResponse)
async def cold_email(
  payload: ColdEmailRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> EmailResponse:
  provider = get_tool_provider(request)
  try:
    result = await email_assistant.generate_cold_email(
      provider,
      company_name=payload.company_name,
      purpose_offer=payload.purpose_offer,
      value_proposition=payload.value_proposition,
      tone=payload.tone,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Cold email generation failed: {exc}") from exc
  return EmailResponse(**result)

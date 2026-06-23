"""AI Email Assistant API."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import email_assistant

router = APIRouter(prefix="/email-assistant", tags=["email-assistant"])


class EmailResponse(BaseModel):
  mode: str
  subject: str
  tone: str
  email: str
  word_count: int


class NewEmailRequest(BaseModel):
  subject: str = Field(default="", max_length=200)
  context: str = Field(..., min_length=1, max_length=3000)
  tone: str | None = Field(default=None, examples=["professional", "friendly", "casual"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])


class ReplyEmailRequest(BaseModel):
  original_email: str = Field(..., min_length=1, max_length=6000)
  reply_points: str = Field(default="", max_length=3000)
  tone: str | None = Field(default=None, examples=["professional", "friendly", "casual"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])


class ColdEmailRequest(BaseModel):
  company_name: str = Field(..., min_length=1, max_length=200)
  purpose_offer: str = Field(..., min_length=1, max_length=1000)
  value_proposition: str = Field(..., min_length=1, max_length=2000)
  tone: str | None = Field(default=None, examples=["professional", "friendly", "persuasive"])
  language: str | None = Field(default=None, examples=["English", "Hindi", "Spanish"])


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
      language=payload.language,
    )
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
      language=payload.language,
    )
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
      language=payload.language,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Cold email generation failed: {exc}") from exc
  return EmailResponse(**result)


"""Post Scheduler API — AI suggestions for post content and hashtags.

Designed to be called directly from the frontend's "AI Suggestions" buttons.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services import post_scheduler
from app.services.registry import ProviderRegistry

router = APIRouter(prefix="/post-scheduler", tags=["post-scheduler"])


class ContentRequest(BaseModel):
  platform: str = Field(..., examples=["instagram", "linkedin", "twitter"])
  topic: str = Field(..., min_length=1, max_length=1000)
  tone: str | None = Field(default=None, examples=["professional", "casual", "funny"])
  keywords: list[str] | None = None
  include_emojis: bool = True
  include_hashtags: bool = False


class ContentResponse(BaseModel):
  platform: str
  content: str
  char_count: int
  char_limit: int


class HashtagRequest(BaseModel):
  platform: str = Field(default="instagram", examples=["instagram", "tiktok"])
  topic: str = Field(..., min_length=1, max_length=2200)
  count: int | None = Field(default=None, ge=1, le=30)


class HashtagResponse(BaseModel):
  platform: str
  hashtags: list[str]
  text: str


class GeneratePostRequest(BaseModel):
  platform: str = Field(..., examples=["instagram", "linkedin"])
  topic: str = Field(..., min_length=1, max_length=1000)
  tone: str | None = None
  keywords: list[str] | None = None
  include_emojis: bool = True
  hashtag_count: int | None = Field(default=None, ge=1, le=30)


class GeneratePostResponse(BaseModel):
  platform: str
  content: str
  char_count: int
  char_limit: int
  hashtags: list[str]
  hashtags_text: str


def _get_provider(request: Request):
  registry: ProviderRegistry = request.app.state.registry
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")
  return registry.provider


@router.get("/platforms")
async def list_platforms(_: str = Depends(verify_api_key)) -> dict:
  return {"platforms": post_scheduler.supported_platforms()}


@router.post("/suggest-content", response_model=ContentResponse)
async def suggest_content(
  payload: ContentRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> ContentResponse:
  provider = _get_provider(request)
  try:
    result = await post_scheduler.suggest_content(
      provider,
      platform=payload.platform,
      topic=payload.topic,
      tone=payload.tone,
      keywords=payload.keywords,
      include_emojis=payload.include_emojis,
      include_hashtags=payload.include_hashtags,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Content generation failed: {exc}") from exc
  return ContentResponse(**result)


@router.post("/suggest-hashtags", response_model=HashtagResponse)
async def suggest_hashtags(
  payload: HashtagRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> HashtagResponse:
  provider = _get_provider(request)
  try:
    result = await post_scheduler.suggest_hashtags(
      provider,
      platform=payload.platform,
      topic=payload.topic,
      count=payload.count,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Hashtag generation failed: {exc}") from exc
  return HashtagResponse(**result)


@router.post("/generate", response_model=GeneratePostResponse)
async def generate_post(
  payload: GeneratePostRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> GeneratePostResponse:
  provider = _get_provider(request)
  try:
    result = await post_scheduler.generate_post(
      provider,
      platform=payload.platform,
      topic=payload.topic,
      tone=payload.tone,
      keywords=payload.keywords,
      include_emojis=payload.include_emojis,
      hashtag_count=payload.hashtag_count,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Post generation failed: {exc}") from exc
  return GeneratePostResponse(**result)

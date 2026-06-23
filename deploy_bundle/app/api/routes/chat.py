import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.security import verify_api_key
from app.services.registry import ProviderRegistry

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
  role: str = Field(..., examples=["user", "assistant", "system"])
  content: str


class ChatCompletionRequest(BaseModel):
  model: str | None = None
  messages: list[ChatMessage]
  max_tokens: int | None = Field(default=None, ge=1, le=2048)
  temperature: float | None = Field(default=None, ge=0.0, le=2.0)
  top_p: float | None = Field(default=None, ge=0.0, le=1.0)
  stream: bool = False


class ChatChoice(BaseModel):
  index: int
  message: ChatMessage
  finish_reason: str


class UsageInfo(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class ChatCompletionResponse(BaseModel):
  id: str
  object: str = "chat.completion"
  created: int
  model: str
  choices: list[ChatChoice]
  usage: UsageInfo


def _get_registry(request: Request) -> ProviderRegistry:
  return request.app.state.registry


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
  payload: ChatCompletionRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> ChatCompletionResponse:
  if payload.stream:
    raise HTTPException(status_code=501, detail="Streaming not yet enabled; set stream=false")

  registry = _get_registry(request)
  if not registry.is_ready():
    raise HTTPException(status_code=503, detail="Model is loading or unavailable")

  messages = [m.model_dump() for m in payload.messages]
  kwargs: dict = {}
  if payload.max_tokens is not None:
    kwargs["max_tokens"] = payload.max_tokens
  if payload.temperature is not None:
    kwargs["temperature"] = payload.temperature
  if payload.top_p is not None:
    kwargs["top_p"] = payload.top_p

  try:
    content = await registry.provider.chat(messages, **kwargs)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

  prompt_chars = sum(len(m["content"]) for m in messages)
  completion_chars = len(content)

  return ChatCompletionResponse(
    id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
    created=int(time.time()),
    model=registry.provider.model_id(),
    choices=[
      ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop",
      )
    ],
    usage=UsageInfo(
      prompt_tokens=max(1, prompt_chars // 4),
      completion_tokens=max(1, completion_chars // 4),
      total_tokens=max(2, (prompt_chars + completion_chars) // 4),
    ),
  )


@router.get("/models")
async def list_models(
  request: Request,
  _: str = Depends(verify_api_key),
) -> dict:
  registry = _get_registry(request)
  model_id = registry.provider.model_id() if registry.is_ready() else "custom-nexus-v1"
  return {
    "object": "list",
    "data": [
      {
        "id": model_id,
        "object": "model",
        "owned_by": "custom",
        "permission": [],
      }
    ],
  }

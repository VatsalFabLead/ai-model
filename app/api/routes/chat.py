import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.security import verify_api_key
from app.services.registry import ProviderRegistry

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
  role: str = Field(..., examples=["user", "assistant", "system"])
  content: str


class ChatCompletionRequest(BaseModel):
  model: str | None = Field(
    default=None,
    description="Backend: custom | ollama | llm | auto. Or prompt prefix: /ollama your question",
  )
  messages: list[ChatMessage]
  max_tokens: int | None = Field(default=None, ge=1, le=2048)
  temperature: float | None = Field(default=None, ge=0.0, le=2.0)
  top_p: float | None = Field(default=None, ge=0.0, le=1.0)
  stream: bool = False
  backend: str | None = Field(
    default=None,
    description="Optional override: custom | ollama | llm | auto",
  )


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
  response: Response,
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
    content, backend_used = await registry.chat(
      messages,
      model=payload.model,
      backend=payload.backend,
      **kwargs,
    )
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

  response.headers["X-Nexus-Backend"] = backend_used

  low = content.lower()
  if "don't have a confident answer" in low and "knowledge.jsonl" in low:
    raise HTTPException(
      status_code=503,
      detail="Server is running an old build. Restart with: python run.py",
    )
  if "_(source: wikipedia" in low:
    content = content.split("_(Source:")[0].strip()

  prompt_chars = sum(len(m["content"]) for m in messages)
  completion_chars = len(content)

  return ChatCompletionResponse(
    id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
    created=int(time.time()),
    model=registry.model_id(backend_used),
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
  settings = get_settings()
  backends = registry.available_backends()
  data: list[dict] = []
  # Primary product model id
  if registry.is_ready():
    data.append({
      "id": registry.model_id("custom"),
      "object": "model",
      "owned_by": "custom",
      "permission": [],
      "description": "Nexus custom transformer — powers all AI tools",
    })
  else:
    data.append({
      "id": settings.model_id,
      "object": "model",
      "owned_by": "custom",
      "permission": [],
    })
  seen = {d["id"] for d in data}
  for b in backends:
    mid = registry.model_id(b)
    if mid not in seen:
      seen.add(mid)
      data.append({
        "id": mid,
        "object": "model",
        "owned_by": b,
        "permission": [],
      })
  data.append({"id": "auto", "object": "model", "owned_by": "router", "permission": []})
  return {"object": "list", "data": data}

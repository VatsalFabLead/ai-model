from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
  registry = request.app.state.registry
  return {
    "status": "ok" if registry.is_ready() else "loading",
    "model_ready": registry.is_ready(),
    "model_id": registry.provider.model_id() if registry.is_ready() else None,
  }


@router.get("/")
async def root() -> dict:
  return {
    "service": "Custom Model API",
    "docs": "/docs",
    "health": "/health",
    "chat": "/v1/chat/completions",
  }

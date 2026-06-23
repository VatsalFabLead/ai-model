"""Mount custom model API into an existing FastAPI/Starlette parent server."""

from fastapi import FastAPI

from app.api.routes import chat, health
from app.config import Settings, get_settings
from app.services.registry import ProviderRegistry


def mount_custom_model(parent: FastAPI, settings: Settings | None = None) -> ProviderRegistry:
  """
  Merge into an existing server:

      from fastapi import FastAPI
      from app.integrate import mount_custom_model

      main_app = FastAPI()
      registry = mount_custom_model(main_app)
  """
  cfg = settings or get_settings()
  registry = ProviderRegistry(cfg)

  @parent.on_event("startup")
  async def _startup() -> None:
    parent.state.registry = registry
    await registry.startup()

  @parent.on_event("shutdown")
  async def _shutdown() -> None:
    await registry.shutdown()

  parent.include_router(health.router)
  parent.include_router(chat.router, prefix=cfg.api_prefix)
  return registry

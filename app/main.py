from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import (
  chat,
  chat_page,
  cover_letter,
  email_assistant,
  health,
  model_test,
  post_scheduler,
  resume_builder,
  schema_markup,
  seo_content,
  seo_keyword,
  seo_optimizer,
  title_meta,
)
from app.config import get_settings
from app.services.registry import ProviderRegistry


def create_app() -> FastAPI:
  settings = get_settings()
  registry = ProviderRegistry(settings)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    app.state.registry = registry
    await registry.startup()
    yield
    await registry.shutdown()

  app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    root_path=settings.root_path,
  )

  limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit] if settings.rate_limit_enabled else [],
    enabled=settings.rate_limit_enabled,
  )
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

  if settings.trusted_hosts != "*":
    hosts = [h.strip() for h in settings.trusted_hosts.split(",") if h.strip()]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  app.include_router(health.router)
  app.include_router(model_test.router)
  app.include_router(chat_page.router)
  app.include_router(chat.router, prefix=settings.api_prefix)
  app.include_router(post_scheduler.router, prefix=settings.api_prefix)
  app.include_router(schema_markup.router, prefix=settings.api_prefix)
  app.include_router(seo_content.router, prefix=settings.api_prefix)
  app.include_router(seo_keyword.router, prefix=settings.api_prefix)
  app.include_router(seo_optimizer.router, prefix=settings.api_prefix)
  app.include_router(title_meta.router, prefix=settings.api_prefix)
  app.include_router(email_assistant.router, prefix=settings.api_prefix)
  app.include_router(resume_builder.router, prefix=settings.api_prefix)
  app.include_router(cover_letter.router, prefix=settings.api_prefix)

  return app


app = create_app()

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
  plagiarism_checker,
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
    import asyncio
    from app.engine.plagiarism_engine import configure, warm_up
    from app.config import get_settings
    cfg = get_settings()
    configure(cfg.plagiarism_index_dir)
    if cfg.plagiarism_warmup_at_start:
      try:
        await asyncio.to_thread(warm_up)
      except Exception as exc:
        import logging
        logging.getLogger("uvicorn.error").warning("Plagiarism warm-up skipped: %s", exc)
    from app.engine.seo_optimizer_rag_pipeline import GENERATOR_VERSION
    import logging
    from app.engine.title_meta_rag_pipeline import GENERATOR_VERSION as TITLE_META_VERSION
    from app.engine.seo_keyword_rag_pipeline import GENERATOR_VERSION as SEO_KEYWORD_VERSION
    from app.engine.resume_rag_pipeline import GENERATOR_VERSION as RESUME_VERSION
    logging.getLogger("uvicorn.error").info("SEO Optimizer pipeline: %s", GENERATOR_VERSION)
    logging.getLogger("uvicorn.error").info("Title & Meta pipeline: %s", TITLE_META_VERSION)
    logging.getLogger("uvicorn.error").info("SEO Keyword pipeline: %s", SEO_KEYWORD_VERSION)
    logging.getLogger("uvicorn.error").info("Resume Builder pipeline: %s", RESUME_VERSION)
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
  app.include_router(plagiarism_checker.router, prefix=settings.api_prefix)
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

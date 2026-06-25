"""Unified Nexus gateway — all AI tools under model custom-nexus-v1."""

from __future__ import annotations

import time
from typing import Any

from app.services import (
  cover_letter,
  email_assistant,
  plagiarism_checker,
  resume_builder,
  schema_markup,
  seo_content,
  seo_keyword,
  seo_optimizer,
  title_meta,
)
from app.services.provider_base import ModelProvider
from app.services.registry import ProviderRegistry

NEXUS_MODEL_ID = "custom-nexus-v1"

# Tool catalog exposed via GET /nexus/tools
NEXUS_TOOL_CATALOG: list[dict[str, Any]] = [
  {"id": "chat", "label": "Chat completions", "endpoint": "/chat/completions"},
  {"id": "seo_content", "label": "SEO Content Generator", "endpoint": "/seo-content/generate"},
  {"id": "seo_optimizer", "label": "SEO Content Optimizer", "endpoint": "/seo-optimizer/optimize"},
  {"id": "title_meta", "label": "SEO Title & Meta", "endpoint": "/title-meta/generate"},
  {"id": "seo_keywords", "label": "SEO Keyword Generator", "endpoint": "/seo-keywords/generate"},
  {"id": "schema_markup", "label": "Schema Markup Generator", "endpoint": "/schema-markup/generate"},
  {"id": "email_new", "label": "Email Assistant — New", "endpoint": "/email-assistant/new-email"},
  {"id": "email_reply", "label": "Email Assistant — Reply", "endpoint": "/email-assistant/reply"},
  {"id": "email_cold", "label": "Email Assistant — Cold", "endpoint": "/email-assistant/cold-email"},
  {"id": "plagiarism_check", "label": "Copyright & Plagiarism Check", "endpoint": "/plagiarism-check/check"},
  {"id": "plagiarism_remove", "label": "Plagiarism Remove & Rewrite", "endpoint": "/plagiarism-check/remove"},
  {"id": "cover_letter", "label": "Professional Cover Letter", "endpoint": "/cover-letter/generate"},
  {"id": "resume_builder", "label": "Resume Builder", "endpoint": "/resume-builder/generate"},
]

_VALID_TOOLS = frozenset(t["id"] for t in NEXUS_TOOL_CATALOG)


def _provider(registry: ProviderRegistry, model: str, *, use_ai: bool = True) -> ModelProvider | None:
  if not use_ai:
    return None
  return registry.get_provider_for_model(model)


async def invoke_nexus_tool(
  registry: ProviderRegistry,
  *,
  tool: str,
  model: str = NEXUS_MODEL_ID,
  input_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Dispatch one tool call; returns structured result dict."""
  tool_id = (tool or "").strip().lower()
  if tool_id not in _VALID_TOOLS:
    raise ValueError(f"Unknown tool '{tool}'. Valid: {', '.join(sorted(_VALID_TOOLS))}")

  inp = dict(input_data or {})
  t0 = time.perf_counter()

  if tool_id == "chat":
    messages = inp.get("messages")
    if not messages:
      raise ValueError("chat requires input.messages")
    kwargs: dict[str, Any] = {}
    for key in ("max_tokens", "temperature", "top_p"):
      if inp.get(key) is not None:
        kwargs[key] = inp[key]
    text, backend_used = await registry.chat(
      messages,
      model=model or NEXUS_MODEL_ID,
      backend=inp.get("backend"),
      **kwargs,
    )
    result = {
      "content": text,
      "backend": backend_used,
      "model": registry.model_id(backend_used),
    }
  elif tool_id == "seo_content":
    use_ai = bool(inp.get("use_ai", True))
    result = await seo_content.generate(
      _provider(registry, model, use_ai=use_ai),
      topic=inp["topic"],
      keywords=inp.get("keywords"),
      tone=inp.get("tone"),
      use_ai=use_ai,
      discover_keywords=bool(inp.get("discover_keywords", False)),
      use_rag=bool(inp.get("use_rag", True)),
      variation_seed=inp.get("variation_seed"),
    )
  elif tool_id == "seo_optimizer":
    use_ai = bool(inp.get("use_ai", True))
    result = await seo_optimizer.optimize(
      _provider(registry, model, use_ai=use_ai),
      content=inp["content"],
      keywords=inp.get("keywords"),
      tone=inp.get("tone"),
      use_ai=use_ai,
      use_rag=bool(inp.get("use_rag", True)),
      variation_seed=inp.get("variation_seed"),
    )
  elif tool_id == "title_meta":
    use_ai = bool(inp.get("use_ai", False))
    result = await title_meta.generate(
      _provider(registry, model, use_ai=use_ai),
      topic=inp["topic"],
      variations=int(inp.get("variations", 10)),
      category=inp.get("category", "blog_article"),
      use_ai=use_ai,
      use_rag=bool(inp.get("use_rag", True)),
      variation_seed=inp.get("variation_seed"),
    )
  elif tool_id == "seo_keywords":
    use_ai = bool(inp.get("use_ai", False))
    result = await seo_keyword.generate_keywords(
      _provider(registry, model, use_ai=use_ai),
      seed_keyword=inp["seed_keyword"],
      variations=int(inp.get("variations", 10)),
      tone=inp.get("tone"),
      use_ai=use_ai,
      use_rag=bool(inp.get("use_rag", True)),
      discover_web=bool(inp.get("discover_web", True)),
      variation_seed=inp.get("variation_seed"),
    )
  elif tool_id == "schema_markup":
    provider = registry.get_provider_for_model(model)
    result = await schema_markup.generate_schema_markup(
      provider,
      schema_type=inp["schema_type"],
      name=inp["name"],
      data=inp.get("data") or {},
      language=inp.get("language"),
      ai_enhance=bool(inp.get("ai_enhance", False)),
      use_rag=bool(inp.get("use_rag", False)),
    )
  elif tool_id == "email_new":
    provider = registry.get_provider_for_model(model)
    result = await email_assistant.generate_new_email(
      provider,
      subject=inp.get("subject", ""),
      context=inp["context"],
      tone=inp.get("tone"),
    )
  elif tool_id == "email_reply":
    provider = registry.get_provider_for_model(model)
    result = await email_assistant.generate_reply_email(
      provider,
      original_email=inp["original_email"],
      reply_points=inp.get("reply_points", ""),
      tone=inp.get("tone"),
    )
  elif tool_id == "email_cold":
    provider = registry.get_provider_for_model(model)
    result = await email_assistant.generate_cold_email(
      provider,
      company_name=inp["company_name"],
      purpose_offer=inp["purpose_offer"],
      value_proposition=inp["value_proposition"],
      tone=inp.get("tone"),
    )
  elif tool_id == "plagiarism_check":
    result = await plagiarism_checker.check_content(content=inp["content"])
  elif tool_id == "plagiarism_remove":
    provider = registry.get_provider_for_model(model)
    result = await plagiarism_checker.remove_plagiarism(provider, content=inp["content"])
  elif tool_id == "cover_letter":
    use_ai = bool(inp.get("use_ai", True))
    result = await cover_letter.generate_cover_letter(
      _provider(registry, model, use_ai=use_ai),
      job_role=inp["job_role"],
      company_name=inp["company_name"],
      skills_experience=inp["skills_experience"],
      tone=inp.get("tone"),
      language=inp.get("language"),
      applicant_name=inp.get("applicant_name"),
      use_ai=use_ai,
      use_rag=bool(inp.get("use_rag", True)),
      variation_seed=inp.get("variation_seed"),
    )
  elif tool_id == "resume_builder":
    use_ai = bool(inp.get("use_ai", True))
    result = await resume_builder.generate(
      _provider(registry, model, use_ai=use_ai),
      full_name=inp["full_name"],
      job_title=inp["job_title"],
      email=inp["email"],
      phone=inp["phone"],
      linkedin=inp.get("linkedin"),
      portfolio=inp.get("portfolio"),
      education=inp.get("education"),
      experience=inp.get("experience"),
      skills=inp.get("skills"),
      summary=inp.get("summary"),
      projects=inp.get("projects"),
      certifications=inp.get("certifications"),
      achievements=inp.get("achievements"),
      languages=inp.get("languages"),
      template=inp.get("template", "modern"),
      template_name=inp.get("template_name"),
      language=inp.get("language"),
      use_ai=use_ai,
      use_rag=bool(inp.get("use_rag", True)),
      variation_seed=inp.get("variation_seed"),
    )
  else:
    raise ValueError(f"Tool not implemented: {tool_id}")

  elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
  return {
    "model": model or NEXUS_MODEL_ID,
    "tool": tool_id,
    "result": result,
    "elapsed_ms": elapsed_ms,
  }

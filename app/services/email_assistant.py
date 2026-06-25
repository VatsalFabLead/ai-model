"""AI Email Assistant — production pipeline v5.2."""

from __future__ import annotations

from typing import Any

from app.engine.email_assistant_pipeline import run_email_assistant_pipeline
from app.services.provider_base import ModelProvider


async def generate_new_email(
  provider: ModelProvider | None,
  *,
  subject: str,
  context: str,
  tone: str | None = None,
) -> dict[str, Any]:
  result = await run_email_assistant_pipeline(
    "new_email",
    {"subject": subject, "context": context, "tone": tone},
    provider,
  )
  return _compact(result)


async def generate_reply_email(
  provider: ModelProvider | None,
  *,
  original_email: str,
  reply_points: str,
  tone: str | None = None,
) -> dict[str, Any]:
  result = await run_email_assistant_pipeline(
    "reply",
    {"original_email": original_email, "reply_points": reply_points, "tone": tone},
    provider,
  )
  return _compact(result)


async def generate_cold_email(
  provider: ModelProvider | None,
  *,
  company_name: str,
  purpose_offer: str,
  value_proposition: str,
  tone: str | None = None,
) -> dict[str, Any]:
  result = await run_email_assistant_pipeline(
    "cold_email",
    {
      "company_name": company_name,
      "purpose_offer": purpose_offer,
      "value_proposition": value_proposition,
      "tone": tone,
    },
    provider,
  )
  return _compact(result)


def _compact(result: dict[str, Any]) -> dict[str, Any]:
  return {
    "mode": result["mode"],
    "subject": result["subject"],
    "tone": result["tone"],
    "email": result["email"],
    "word_count": result["word_count"],
    "subject_options": result.get("subject_options", []),
    "quality": result.get("quality", {}),
    "scores": result.get("scores", {}),
    "reading_time_minutes": result.get("reading_time_minutes"),
    "suggestions": result.get("suggestions", []),
    "alternatives": result.get("alternatives", []),
    "generator_version": result.get("generator_version"),
    "language": result.get("language"),
    "output_language": result.get("output_language"),
    "architecture": result.get("architecture"),
    "pipeline": result.get("pipeline"),
    "ai": result.get("ai"),
    "elapsed_ms": result.get("elapsed_ms"),
  }

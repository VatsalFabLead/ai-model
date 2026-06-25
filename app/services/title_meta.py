"""SEO Title & Meta Description Generator — RAG pipeline + optional local model polish."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.engine import title_meta_engine as engine
from app.engine.title_meta_rag_pipeline import run_title_meta_pipeline
from app.services.provider_base import ModelProvider

TITLE_MAX = engine.TITLE_MAX
META_MIN = engine.META_MIN
META_MAX = engine.META_MAX


def supported_categories() -> list[dict[str, str]]:
  return engine.supported_categories()


def supported_tones() -> list[dict[str, str]]:
  return engine.supported_tones()


def supported_languages() -> list[dict[str, str]]:
  return engine.supported_languages()


def _parse_variations(raw: str, topic: str, count: int) -> list[dict]:
  text = (raw or "").strip()
  variations: list[dict] = []
  if text.startswith("["):
    try:
      arr = json.loads(text)
      if isinstance(arr, list):
        for item in arr[:count]:
          if isinstance(item, dict):
            title = str(item.get("title", ""))
            meta = str(item.get("meta_description") or item.get("meta") or "")
            if title and meta:
              variations.append({"title": title, "meta_description": meta, "angle": "ai"})
        return variations
    except Exception:
      pass
  blocks = re.split(r"(?:\n\s*---\s*\n)", text, flags=re.IGNORECASE)
  for block in blocks:
    m = re.search(r"TITLE\s*[:\-]\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
    m2 = re.search(r"META(?:\s*DESCRIPTION)?\s*[:\-]\s*(.+)", block, re.IGNORECASE)
    if m and m2:
      variations.append({
        "title": m.group(1).strip(),
        "meta_description": m2.group(1).strip().split("\n")[0],
        "angle": "ai",
      })
    if len(variations) >= count:
      break
  return variations


async def generate(
  provider: ModelProvider | None,
  *,
  topic: str,
  variations: int = 10,
  tone: str | None = None,
  language: str | None = None,
  category: str | None = None,
  use_ai: bool = True,
  use_rag: bool = True,
  variation_seed: int | None = None,
) -> dict:
  topic = re.sub(r"\s+", " ", (topic or "").strip())
  if not topic:
    raise ValueError("topic is required")

  n = max(10, min(50, variations))
  cat = engine.normalize_category(category)
  tone_str = engine.normalize_tone(tone, cat)
  lang_code = engine.bcp47(language)
  if variation_seed is None:
    variation_seed = int(time.time() * 1000) % 2_000_000_000

  ai_used = False
  result = await run_title_meta_pipeline(
    topic,
    variations=n,
    tone=tone_str,
    category=cat,
    variation_seed=variation_seed,
    use_rag=use_rag,
  )

  items: list[dict[str, Any]] = list(result["variations"])
  policy = result.get("policy") or {}

  if use_ai and provider is not None and len(items) < n and not policy.get("safe_mode"):
    lang_line = f" Write in {language}." if language else ""
    try:
      raw = await provider.chat(
        [{"role": "user", "content": f"Topic: {topic}. Generate {min(3, n)} unique title+meta pairs."}],
        system_prompt=(
          f"SEO copywriter ({tone_str}). Title max {TITLE_MAX}, meta {META_MIN}-{META_MAX}.{lang_line} "
          "Format: TITLE: ...\\nMETA: ...\\n---"
        ),
        use_rag=False,
        skip_intent=True,
        max_tokens=400,
        temperature=0.8,
      )
      for extra in _parse_variations(raw, topic, n - len(items)):
        q = engine.quality_variation(extra["title"], extra["meta_description"], topic)
        extra.update({
          "title_length": len(extra["title"]),
          "meta_length": len(extra["meta_description"]),
          "quality_score": q["quality_score"],
          "seo_ready": q["seo_ready"],
          "issues": q["issues"],
        })
        existing = {v["title"].lower() for v in items}
        if extra["title"].lower() not in existing:
          items.append(extra)
      ai_used = True
    except Exception:
      pass

  items = items[:n]
  avg_quality = int(round(sum(v.get("quality_score", 0) for v in items) / max(len(items), 1)))

  return {
    "topic": topic,
    "category": cat,
    "language": lang_code,
    "tone": tone_str,
    "variations": items,
    "variation_count": len(items),
    "title_limit": TITLE_MAX,
    "meta_min": META_MIN,
    "meta_max": META_MAX,
    "quality": {
      "average_score": avg_quality,
      "seo_ready": avg_quality >= 75,
      "all_ready": all(v.get("seo_ready") for v in items),
    },
    "ai": {"enabled": use_ai, "model_used": ai_used},
    "generator_version": result.get("generator_version"),
    "variation_seed": result.get("variation_seed"),
    "policy": policy,
    "architecture": result.get("architecture"),
    "pipeline": result.get("pipeline"),
    "rag": result.get("rag"),
    "elapsed_ms": result.get("elapsed_ms"),
    "unlimited_outputs": result.get("unlimited_outputs"),
  }


async def generate_title_meta(provider: ModelProvider | None, **kwargs) -> dict:
  return await generate(provider, **kwargs)

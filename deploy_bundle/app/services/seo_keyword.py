"""SEO Keyword Generator — RAG pipeline + optional local model enrichment."""

from __future__ import annotations

import re
from typing import Any

from app.engine.seo_keyword_rag_pipeline import run_seo_keyword_pipeline
from app.services.provider_base import ModelProvider

_VALID_TONES = {
  "professional", "casual", "friendly", "funny", "excited", "inspirational",
  "bold", "informative", "persuasive", "authoritative", "conversational", "neutral",
}


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "informative"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "informative"


def _parse_ai_lines(raw: str, seed_keyword: str, max_items: int) -> list[str]:
  lines = [re.sub(r"^\d+[\).\-\s]+", "", ln).strip().lower() for ln in (raw or "").splitlines()]
  cleaned: list[str] = []
  for ln in lines:
    ln = _clean(re.sub(r"^[#\-\*\•]+\s*", "", ln))
    if not ln or len(ln) < 3 or len(ln) > 80:
      continue
    cleaned.append(ln)
  seen: set[str] = set()
  out: list[str] = []
  for k in cleaned:
    if k not in seen:
      seen.add(k)
      out.append(k)
    if len(out) >= max_items:
      break
  return out


async def generate_keywords(
  provider: ModelProvider | None,
  *,
  seed_keyword: str,
  tone: str | None = None,
  max_items: int = 10,
  variations: int | None = None,
  language: str | None = None,
  use_ai: bool = False,
  use_rag: bool = True,
  discover_web: bool = True,
  include_questions: bool = True,
  include_alphabet: bool = True,
  variation_seed: int | None = None,
) -> dict[str, Any]:
  seed_keyword = _clean(seed_keyword)
  if not seed_keyword:
    raise ValueError("seed_keyword is required")

  n = max(10, min(50, variations if variations is not None else max_items))
  tone_str = _normalize_tone(tone)

  result = await run_seo_keyword_pipeline(
    seed_keyword,
    variations=n,
    tone=tone_str,
    language=language,
    variation_seed=variation_seed,
    use_rag=use_rag,
    discover_web=discover_web,
    include_questions=include_questions,
    include_alphabet=include_alphabet,
  )

  items: list[dict[str, Any]] = list(result["keywords"])

  if use_ai and provider is not None:
    preview = ", ".join(k["keyword"] for k in items[:10])
    try:
      raw = await provider.chat(
        [{"role": "user", "content": f"Seed: {seed_keyword}. Add {min(5, n)} unique SEO keywords not in: {preview}"}],
        system_prompt=(
          "SEO keyword researcher. Return plain lines only, one keyword per line, no numbering."
        ),
        use_rag=False,
        skip_intent=True,
        max_tokens=300,
        temperature=0.75,
      )
      from app.engine.seo_keyword_rag_pipeline import build_keyword_row

      seed = result.get("variation_seed") or 0
      existing = {k["keyword"].lower() for k in items}
      for i, kw in enumerate(_parse_ai_lines(raw, seed_keyword, 5)):
        if kw.lower() not in existing:
          row = build_keyword_row(
            kw, seed=seed_keyword, sources=["ai"], relevance=50, variation_seed=seed + 900 + i,
          )
          items.append(row)
          existing.add(kw.lower())
      items = items[:n]
      result["ai"] = {"enabled": True, "model_used": True}
    except Exception:
      result["ai"] = {"enabled": True, "model_used": False}
  else:
    result["ai"] = {"enabled": use_ai, "model_used": False}

  result["keywords"] = items[:n]
  result["count"] = len(result["keywords"])
  return result

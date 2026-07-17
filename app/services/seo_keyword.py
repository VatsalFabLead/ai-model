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


def _extract_seed_from_context(context: str) -> str:
  """Pull a short seed topic from a longer business/context brief."""
  text = (context or "").strip()
  if not text:
    return ""
  # Prefer quoted or labeled topic
  m = re.search(
    r"(?:seed|keyword|topic|brand|product|about)\s*[:=\-]\s*[\"']?([^\"'\n,]{2,80})",
    text,
    re.I,
  )
  if m:
    return _clean(m.group(1))
  # Prefer strong product/industry noun phrases
  low = text.lower()
  for phrase in (
    "coffee roastery", "specialty coffee", "coffee shop", "coffee brand",
    "real estate", "digital marketing", "seo agency", "saas platform",
    "online store", "ecommerce store", "fitness studio", "yoga studio",
  ):
    if phrase in low:
      return phrase
  # First sentence / line, truncated to a usable seed
  first = re.split(r"[.\n!?]", text, maxsplit=1)[0]
  first = _clean(first)
  first = re.sub(
    r"^(?:we\s+(?:are|run|sell|offer|provide)|our\s+(?:company|brand|product)\s+is)\s+",
    "",
    first,
    flags=re.I,
  )
  first = re.sub(r"^(?:a|an|the)\s+", "", first, flags=re.I)
  words = first.split()
  # Prefer noun-ish window: drop filler, keep 3–6 content words
  skip = {"in", "at", "to", "for", "and", "with", "from", "of", "the", "a", "an", "selling", "offering"}
  content = [w for w in words if w.lower() not in skip]
  if len(content) >= 2:
    first = " ".join(content[:5])
  elif len(words) > 6:
    first = " ".join(words[:6])
  return first[:80].strip(" ,;-")


def resolve_seed_and_context(
  seed_keyword: str | None,
  context: str | None,
) -> tuple[str, str]:
  """Return (seed, context_brief). Context alone is enough."""
  seed = _clean(seed_keyword)
  brief = (context or "").strip()
  if not seed and brief:
    seed = _extract_seed_from_context(brief) or brief[:80]
  if not seed and not brief:
    raise ValueError("Provide seed_keyword or context (business / topic brief)")
  if not brief and seed:
    brief = seed
  # If seed is actually a long brief pasted into seed_keyword, treat as context
  if seed and not context and (len(seed) > 120 or seed.count(" ") >= 18):
    brief = seed
    seed = _extract_seed_from_context(seed) or " ".join(seed.split()[:6])
  return seed, brief


async def generate_keywords(
  provider: ModelProvider | None,
  *,
  seed_keyword: str = "",
  context: str | None = None,
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
  seed, brief = resolve_seed_and_context(seed_keyword, context)

  n = max(10, min(50, variations if variations is not None else max_items))
  tone_str = _normalize_tone(tone)

  result = await run_seo_keyword_pipeline(
    seed,
    context_text=brief,
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
  entities_detected = (
    result.get("pipeline", {}).get("entities")
    or result.get("architecture", {}).get("stages", {}).get("named_entity_recognition", {}).get("entities")
  )

  if use_ai and provider is not None and (entities_detected or brief):
    preview = ", ".join(k["keyword"] for k in items[:10])
    primary = (result.get("pipeline") or {}).get("primary_domain") or ""
    try:
      raw = await provider.chat(
        [{
          "role": "user",
          "content": (
            f"Seed: {seed}\n"
            f"Primary domain: {primary}\n"
            f"Context: {brief[:1200]}\n"
            f"Add {min(8, n)} unique SEO keywords not already in: {preview}\n"
            "Stay strictly on this business topic. Do NOT invent healthcare, "
            "AI agency, software development, or unrelated industry keywords."
          ),
        }],
        system_prompt=(
          "SEO keyword researcher. Keywords must match the user's context only. "
          "Reject off-topic verticals. Return plain lines only, one keyword per line."
        ),
        use_rag=False,
        skip_intent=True,
        max_tokens=400,
        temperature=0.55,
      )
      from app.engine.seo_keyword_rag_pipeline import build_keyword_row
      from app.engine.seo_keyword_relevance import passes_topic_relevance

      ctx = dict(result.get("pipeline", {}).get("context") or {})
      ctx.setdefault("seed_keyword", seed)
      ctx.setdefault("context_brief", brief)
      ctx.setdefault("primary_domain", primary)
      added = 0
      for kw in _parse_ai_lines(raw, seed, min(8, n)):
        if any(it["keyword"] == kw for it in items):
          continue
        if not passes_topic_relevance(kw, ctx, min_score=28):
          continue
        items.append(build_keyword_row(
          kw, context=ctx, sources=["ai_enrichment"], relevance=78, variation_seed=0,
        ))
        added += 1
      backend = getattr(provider, "last_backend", None) or getattr(provider, "model_id", None)
      result["ai"] = {
        "enabled": True,
        "model_used": True,
        "backend": backend,
        "added": added,
        "hosted": str(backend or "").lower() in ("hosted", "groq") or "hosted" in str(type(provider).__name__).lower(),
      }
    except Exception:
      result["ai"] = {"enabled": True, "model_used": False, "hosted": False}
  elif use_ai and provider is not None:
    result["ai"] = {"enabled": use_ai, "model_used": False, "hosted": False}
  else:
    result["ai"] = {"enabled": False, "model_used": False, "hosted": False}

  result["keywords"] = items[:n]
  result["count"] = len(result["keywords"])
  result["context"] = brief if brief != seed else None
  return result

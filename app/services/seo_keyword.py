"""SEO Keyword Generator — content strategist (hosted) + RAG pipeline fallback."""

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
  m = re.search(
    r"(?:seed|keyword|topic|brand|product|about)\s*[:=\-]\s*[\"']?([^\"'\n,]{2,80})",
    text,
    re.I,
  )
  if m:
    return _clean(m.group(1))
  low = text.lower()
  for phrase in (
    "coffee roastery", "specialty coffee", "coffee shop", "coffee brand",
    "real estate", "digital marketing", "seo agency", "saas platform",
    "online store", "ecommerce store", "fitness studio", "yoga studio",
  ):
    if phrase in low:
      return phrase
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
    raise ValueError("Provide seed_keyword, context, title, or content")
  if not brief and seed:
    brief = seed
  if seed and not context and (len(seed) > 120 or seed.count(" ") >= 18):
    brief = seed
    seed = _extract_seed_from_context(seed) or " ".join(seed.split()[:6])
  return seed, brief


def _should_use_strategist(
  *,
  use_ai: bool,
  provider: ModelProvider | None,
  title: str,
  content: str,
  context: str,
  mode: str | None,
) -> bool:
  if not use_ai or provider is None:
    return False
  if (mode or "").lower() in ("pipeline", "rag", "legacy"):
    return False
  if (mode or "").lower() in ("strategist", "content", "article"):
    return True
  # Prefer strategist whenever hosted AI is on and we have real content/title
  if title or len(content) >= 40 or len(context) >= 40:
    return True
  return False


async def generate_keywords(
  provider: ModelProvider | None,
  *,
  seed_keyword: str = "",
  context: str | None = None,
  title: str | None = None,
  content: str | None = None,
  primary_topic: str | None = None,
  country: str | None = None,
  market: str | None = None,
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
  mode: str | None = None,
) -> dict[str, Any]:
  title_s = _clean(title)
  content_s = (content or "").strip()
  topic_s = _clean(primary_topic) or _clean(seed_keyword)
  country_s = _clean(country) or _clean(market)
  brief_ctx = (context or "").strip()

  # Merge: content field wins; else context; else seed-as-brief
  if not content_s and brief_ctx:
    content_s = brief_ctx
  if not title_s and topic_s:
    title_s = topic_s

  n = max(10, min(50, variations if variations is not None else max_items))

  if _should_use_strategist(
    use_ai=use_ai,
    provider=provider,
    title=title_s,
    content=content_s,
    context=brief_ctx,
    mode=mode,
  ):
    from app.engine.seo_keyword_strategist import generate_with_strategist

    try:
      return await generate_with_strategist(
        provider,  # type: ignore[arg-type]
        title=title_s,
        content=content_s or brief_ctx or topic_s,
        primary_topic=topic_s or title_s,
        country=country_s,
        language=language or "English",
        max_keywords=n,
      )
    except Exception:
      # Fall through to deterministic pipeline if hosted strategist fails
      pass

  seed, brief = resolve_seed_and_context(
    seed_keyword or topic_s or title_s,
    content_s or brief_ctx or None,
  )
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
  output_sec = result.get("output", {})
  if isinstance(output_sec, dict):
    result["cannibalization_warnings"] = output_sec.get("cannibalization_warnings")
    result["export_bundle"] = output_sec.get("export_bundle")
  if title_s:
    result["title"] = title_s
  if topic_s:
    result["primary_topic"] = topic_s
  if country_s:
    result["market"] = country_s
  return result

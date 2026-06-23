"""SEO Content Generator — advanced, multilingual, worldwide.

Structured output: metadata, keywords, outline, content (article + tone), FAQs.
Template-first for speed; optional custom-model polish when use_ai=True. No GPT/Claude/Gemini.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.engine import seo_content_engine
from app.engine.keyword_discovery import discover_keywords
from app.engine.seo_content_domains import build_rich_content, make_variation_seed
from app.engine.seo_rag_pipeline import run_seo_rag_pipeline, synthesize_structured_content
from app.services.provider_base import ModelProvider

_GENERIC_HEADINGS = {
  "introduction", "intro", "overview", "summary", "contents", "table of contents",
  "getting started", "background", "conclusion", "about",
}

_AI_TIMEOUT_SEC = 14.0
_SPAM_PHRASES = (
  "search visibility", "compounding traffic", "keyword stuffing",
  "marketers, business owners, creators", "complete 2026 guide",
  "gain visibility, trust, and sustainable growth",
)


def _is_spam_content(text: str) -> bool:
  low = (text or "").lower()
  return sum(1 for p in _SPAM_PHRASES if p in low) >= 2


def supported_categories() -> list[dict[str, str]]:
  return seo_content_engine.supported_categories()


def supported_tones() -> list[dict[str, str]]:
  return seo_content_engine.supported_tones()


def supported_languages() -> list[dict[str, str]]:
  return seo_content_engine.supported_languages()


def coerce_keywords(keywords: list[str] | str | None) -> list[str]:
  if not keywords:
    return []
  if isinstance(keywords, str):
    parts = keywords.split(",")
  else:
    parts = [str(k) for k in keywords]
  seen: set[str] = set()
  out: list[str] = []
  for p in parts:
    p = p.strip()
    if p and p.lower() not in seen:
      seen.add(p.lower())
      out.append(p)
  return out


def _slugify(text: str, max_len: int = 60) -> str:
  text = (text or "").lower().strip()
  text = re.sub(r"[^a-z0-9\s-]", "", text)
  text = re.sub(r"[\s_-]+", "-", text).strip("-")
  if len(text) > max_len:
    text = text[:max_len].rsplit("-", 1)[0]
  return text or "untitled"


def _count_words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _clean_body(body: str) -> str:
  cleaned: list[str] = []
  for ln in (body or "").split("\n"):
    m = re.match(r"^\s*(#{1,6})\s*(.*)$", ln)
    if m:
      level, rest = m.group(1), m.group(2)
      rest = re.sub(r"#{1,6}", "", rest).replace("**", "").strip().strip("*_`").strip()
      if not rest:
        continue
      cleaned.append(f"{level} {rest}")
    else:
      cleaned.append(ln)
  return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def _trim_meta(meta: str, limit: int = 160) -> str:
  meta = re.sub(r"\s+", " ", (meta or "").strip()).strip('"\u201c\u201d')
  if len(meta) <= limit:
    return meta
  return meta[: limit - 3].rsplit(" ", 1)[0].rstrip() + "..."


def _try_json(text: str) -> dict | None:
  t = (text or "").strip()
  if not t:
    return None
  if not t.startswith("{"):
    m = re.search(r"\{[\s\S]*\}", t)
    t = m.group(0) if m else t
  try:
    obj = json.loads(t)
    return obj if isinstance(obj, dict) else None
  except Exception:
    return None


def _coerce_faqs(raw: Any) -> list[dict[str, str]]:
  if not isinstance(raw, list):
    return []
  out: list[dict[str, str]] = []
  for item in raw:
    if isinstance(item, dict):
      q = str(item.get("question") or item.get("q") or "").strip()
      a = str(item.get("answer") or item.get("a") or "").strip()
      if q:
        out.append({"question": q, "answer": a})
    elif isinstance(item, str) and item.strip():
      out.append({"question": item.strip(), "answer": ""})
  return out


def _coerce_outline(raw: Any) -> list[str]:
  if not isinstance(raw, list):
    return []
  return [str(x).strip() for x in raw if str(x).strip()]


def _coerce_keywords_struct(raw: Any, fallback: list[str]) -> dict[str, Any]:
  if isinstance(raw, dict) and raw.get("primary"):
    sec = raw.get("secondary") or []
    return {
      "primary": str(raw["primary"]).strip(),
      "secondary": [str(s).strip() for s in sec if str(s).strip()],
    }
  if isinstance(raw, list) and raw:
    return {"primary": str(raw[0]).strip(), "secondary": [str(x).strip() for x in raw[1:] if str(x).strip()]}
  if fallback:
    return {"primary": fallback[0], "secondary": fallback[1:]}
  return {"primary": "", "secondary": []}


def _coerce_outline_struct(raw: Any) -> list[dict[str, str]]:
  if not isinstance(raw, list):
    return []
  out: list[dict[str, str]] = []
  for item in raw:
    if isinstance(item, dict) and item.get("text"):
      level = str(item.get("level") or "h2").lower()
      if level not in ("h1", "h2", "h3"):
        level = "h2"
      out.append({"level": level, "text": str(item["text"]).strip()})
    elif isinstance(item, str) and item.strip():
      out.append({"level": "h2", "text": item.strip()})
  return out


def _outline_to_strings(outline: list[dict[str, str]]) -> list[str]:
  return [o["text"] for o in outline]


def _build_template_structured(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  tone: str,
  audience: str | None,
  language: str | None,
  variation_seed: int,
) -> dict[str, Any]:
  return build_rich_content(
    topic,
    keywords,
    category=category,
    tone=tone,
    audience=audience,
    seed=variation_seed,
  )


def _parse_structured_ai(raw: str, topic: str) -> dict[str, Any] | None:
  obj = _try_json(raw)
  if not obj:
    return None

  meta_obj = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
  content_obj = obj.get("content") if isinstance(obj.get("content"), dict) else {}

  title = (
    meta_obj.get("title") or obj.get("title") or ""
  ).strip()
  meta = (
    meta_obj.get("meta_description") or obj.get("meta_description") or obj.get("meta") or ""
  ).strip()
  article = (
    content_obj.get("article") or obj.get("article") or obj.get("content") or obj.get("body") or ""
  ).strip()
  tone = (content_obj.get("tone") or obj.get("tone") or "").strip()

  outline = _coerce_outline_struct(obj.get("outline"))
  faqs = _coerce_faqs(obj.get("faqs"))
  keywords = _coerce_keywords_struct(obj.get("keywords"), [])

  if not article or _count_words(article) < 60:
    return None

  if not title:
    title = topic.strip().title()[:70]
  if not meta:
    meta = _trim_meta(article[:200])

  article = seo_content_engine.strip_faq_section(_clean_body(article))
  if not outline:
    extracted = seo_content_engine.extract_outline_from_body(article)
    outline = [{"level": "h2", "text": t} for t in extracted]
  if not faqs:
    faqs = seo_content_engine.extract_faqs_from_body(obj.get("content") or article)
  if not keywords.get("primary"):
    keywords = {"primary": topic, "secondary": []}

  return {
    "metadata": {"title": title, "meta_description": _trim_meta(meta)},
    "keywords": keywords,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
  }


def _merge_ai_into_template(template: dict[str, Any], ai: dict[str, Any]) -> dict[str, Any]:
  merged = {
    "metadata": dict(template["metadata"]),
    "keywords": dict(template["keywords"]) if isinstance(template.get("keywords"), dict) else {"primary": "", "secondary": []},
    "outline": list(template["outline"]),
    "content": dict(template["content"]),
    "faqs": list(template["faqs"]),
  }
  if not merged["keywords"].get("primary"):
    merged["keywords"] = _coerce_keywords_struct(template.get("keywords"), [])
  if ai["metadata"].get("title"):
    merged["metadata"]["title"] = ai["metadata"]["title"]
  if ai["metadata"].get("meta_description"):
    merged["metadata"]["meta_description"] = ai["metadata"]["meta_description"]
  if ai["content"].get("article") and _count_words(ai["content"]["article"]) >= 60:
    merged["content"]["article"] = ai["content"]["article"]
  if ai["content"].get("tone"):
    merged["content"]["tone"] = ai["content"]["tone"]
  if ai.get("keywords") and isinstance(ai["keywords"], dict) and ai["keywords"].get("primary"):
    seen = {merged["keywords"]["primary"].lower()}
    for s in merged["keywords"].get("secondary", []):
      seen.add(s.lower())
    for kw in ai["keywords"].get("secondary", []):
      if kw.lower() not in seen:
        merged["keywords"]["secondary"].append(kw)
        seen.add(kw.lower())
  if ai.get("outline"):
    merged["outline"] = ai["outline"]
  if ai.get("faqs"):
    merged["faqs"] = ai["faqs"]
  return merged


def _pack_response(
  structured: dict[str, Any],
  *,
  topic: str,
  category: str,
  lang_code: str,
  discovery_meta: dict[str, Any],
  use_ai: bool,
  ai_used: bool,
) -> dict[str, Any]:
  meta = structured["metadata"]
  article = structured["content"]["article"]
  tone = structured["content"]["tone"]
  kw_struct = _coerce_keywords_struct(structured.get("keywords"), [])
  primary = kw_struct["primary"]
  secondary = kw_struct["secondary"]
  keywords_flat = [primary] + secondary if primary else secondary
  outline_struct = _coerce_outline_struct(structured.get("outline"))
  title = meta["title"]
  meta_desc = meta["meta_description"]

  quality = seo_content_engine.quality_report(title, meta_desc, article, keywords_flat)
  return {
    "topic": topic,
    "category": category,
    "language": lang_code,
    "metadata": meta,
    "keywords": kw_struct,
    "keywords_list": keywords_flat,
    "outline": outline_struct,
    "outline_text": _outline_to_strings(outline_struct),
    "content": {"article": article, "tone": tone},
    "article": article,
    "faqs": structured["faqs"],
    "tone": tone,
    "title": title,
    "meta_description": meta_desc,
    "slug": _slugify(title),
    "word_count": _count_words(article),
    "quality": quality,
    "discovery": discovery_meta,
    "ai": {"enabled": use_ai, "model_used": ai_used},
    "variation_seed": structured.get("variation_seed"),
    "domain": structured.get("domain"),
    "generator_version": "seo-rag-v2",
  }


async def _enhance_with_ai(
  provider: ModelProvider,
  template: dict[str, Any],
  *,
  topic: str,
  primary: str,
  kw_line: str,
  category: str,
  tone: str,
  target: int,
  audience_line: str,
  lang_line: str,
  structure: str,
  evidence_context: str = "",
) -> dict[str, Any] | None:
  tone_guide = seo_content_engine.tone_hint(tone)
  system_prompt = (
    f"You are an expert content writer ({tone} — {tone_guide}). "
    f"Write REAL, topic-specific content about the subject — not generic SEO/marketing advice. "
    f"Category: {category.replace('_', ' ')} (~{target} words).{lang_line}{audience_line} "
    f"Structure: {structure}. "
    "Return ONLY valid JSON:\n"
    '{"metadata":{"title":"...","meta_description":"..."},'
    '"keywords":{"primary":"...","secondary":["..."]},'
    '"outline":[{"level":"h1|h2|h3","text":"..."}],'
    '"content":{"article":"markdown article WITHOUT FAQ section","tone":"' + tone + '"},'
    '"faqs":[{"question":"...","answer":"..."}]}'
  )
  draft = json.dumps(template, ensure_ascii=False)[:2800]
  evidence_block = (evidence_context or "")[:2400]
  user_prompt = (
    f"Topic: {topic}\nPrimary keyword: {primary}\nKeywords: {kw_line}\n"
    f"Evidence from open datasets (use facts, do not invent):\n{evidence_block}\n\n"
    f"Rewrite into polished SEO JSON (unique wording, topic-specific):\n{draft}"
  )
  max_tokens = min(900, int(target * 1.2) + 120)
  raw = await asyncio.wait_for(
    provider.chat(
      [{"role": "user", "content": user_prompt}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=max_tokens,
      temperature=0.55,
    ),
    timeout=_AI_TIMEOUT_SEC,
  )
  return _parse_structured_ai(raw, topic)


async def generate(
  provider: ModelProvider | None,
  *,
  topic: str,
  keywords: list[str] | str | None = None,
  tone: str | None = None,
  word_count: int | None = None,
  audience: str | None = None,
  category: str | None = None,
  language: str | None = None,
  use_ai: bool = True,
  discover_keywords: bool = False,
  max_keyword_items: int = 10,
  variation_seed: int | None = None,
  use_rag: bool = True,
) -> dict[str, Any]:
  """Structured SEO content — open-data RAG + optional custom-model synthesis."""
  topic = (topic or "").strip()
  if not topic:
    raise ValueError("topic is required")

  cat = seo_content_engine.normalize_category(category)
  tone_str = seo_content_engine.normalize_tone(tone, cat)
  lang_code = seo_content_engine.bcp47(language)
  target = max(150, min(1500, word_count or 500))

  kws = coerce_keywords(keywords)
  discovery_meta: dict[str, Any] = {
    "enabled": discover_keywords,
    "sources_used": [],
    "keyword_count": 0,
  }

  if discover_keywords:
    seed = kws[0] if kws else topic
    disc = await discover_keywords(seed, language=language, include_alphabet=False)
    discovered = [d["keyword"] for d in disc.get("keywords", [])[:max_keyword_items]]
    discovery_meta["sources_used"] = disc.get("sources_used", [])
    discovery_meta["keyword_count"] = len(discovered)
    for kw in discovered:
      if kw.lower() not in {x.lower() for x in kws}:
        kws.append(kw)
    if not kws:
      kws = discovered[:5]

  primary = kws[0] if kws else topic
  kw_line = ", ".join(kws) if kws else topic
  audience_line = f" Target audience: {audience.strip()}." if audience else ""
  lang_line = f" Write in {language} ({lang_code})." if language else ""
  structure = seo_content_engine.category_structure_hint(cat)
  seed = make_variation_seed(variation_seed)

  rag_meta: dict[str, Any] = {"enabled": use_rag, "confidence": 0.0, "sources_used": []}
  evidence_context = ""

  if use_rag:
    try:
      rag = await run_seo_rag_pipeline(
        topic, kws, category=cat, variation_seed=seed, top_k=8,
      )
      rag_meta = {
        "enabled": True,
        "topic_class": rag.topic_class,
        "confidence": rag.confidence,
        "sources_routed": rag.sources_routed,
        "sources_used": rag.sources_used,
        "document_count": len(rag.documents),
        "fact_count": len(rag.facts),
        "entities": rag.entities[:10],
        "variation_seed": rag.variation_seed,
      }
      evidence_context = rag.evidence_context
      template = synthesize_structured_content(
        topic, kws, rag,
        category=cat, tone=tone_str, audience=audience, target_words=target,
      )
    except Exception:
      template = _build_template_structured(
        topic, kws, category=cat, tone=tone_str, audience=audience,
        language=language, variation_seed=seed,
      )
  else:
    template = _build_template_structured(
      topic, kws, category=cat, tone=tone_str, audience=audience,
      language=language, variation_seed=seed,
    )

  ai_used = False
  structured = template

  if use_ai and provider is not None:
    try:
      ai_result = await _enhance_with_ai(
        provider,
        template,
        topic=topic,
        primary=primary,
        kw_line=kw_line,
        category=cat,
        tone=tone_str,
        target=target,
        audience_line=audience_line,
        lang_line=lang_line,
        structure=structure,
        evidence_context=evidence_context,
      )
      if ai_result and not _is_spam_content(ai_result.get("content", {}).get("article", "")):
        structured = _merge_ai_into_template(template, ai_result)
        ai_used = True
    except Exception:
      structured = template

  result = _pack_response(
    structured,
    topic=topic,
    category=cat,
    lang_code=lang_code,
    discovery_meta=discovery_meta,
    use_ai=use_ai,
    ai_used=ai_used,
  )
  result["rag"] = rag_meta
  return result


# Backward-compatible alias
async def generate_seo_content(provider: ModelProvider | None, **kwargs) -> dict:
  return await generate(provider, **kwargs)

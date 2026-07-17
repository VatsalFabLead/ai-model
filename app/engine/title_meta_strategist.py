"""Elite SEO Title & Meta Strategist — hosted LLM, page-type aware.

Not a template generator. Titles/metas must match detected page type and intent.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.engine import title_meta_engine as engine
from app.services.provider_base import ModelProvider

STRATEGIST_VERSION = "title-meta-strategist-v1"
TITLE_SOFT_MAX = 60
TITLE_HARD_MAX = 65
META_MIN = 145
META_MAX = 160

_BANNED_TITLE_FRAGMENTS = (
  "complete guide",
  "everything you need to know",
  "ultimate guide",
  "best practices",
  "checklist",
)

_GUIDE_OK_TYPES = {
  "blog",
  "educational guide",
  "tutorial",
  "documentation",
  "faq",
}

_SYSTEM = """You are an Elite SEO Title & Meta Description Strategist.

Your job is to generate high-performing SEO titles and meta descriptions that maximize:
• Google rankings
• Organic CTR
• Semantic relevance
• AI Search visibility
• User engagement

You are NOT a template generator.
You are an SEO strategist.

## PRIMARY OBJECTIVE
Generate titles and meta descriptions that accurately represent the page.

Never force these phrases unless they naturally match the page:
• Complete Guide
• Best Practices
• Everything You Need to Know
• Ultimate Guide
• Checklist
• Tips
• Strategies

## PAGE TYPE CONTROLS TITLES
Classify exactly one:
Homepage | Service Page | Local Service Page | Landing Page | Product Page | Category Page | Blog | Educational Guide | Tutorial | Case Study | Brand Story | News | Comparison | Review | Documentation | FAQ

Examples:
• Guide / Blog / Tutorial → educational titles OK
• Service Page → service-focused ("Professional Moving Business Services") NOT "Moving Business Complete Guide"
• Homepage → brand-focused
• Product Page → product-focused — NOT blog/guide titles
• Case Study → outcome-focused
• Brand Story → narrative titles
• News → NOT guide titles
• Local SEO / Local Service Page → include location naturally

## TITLE RULES
Generate 5–10 unique titles (or up to the requested count).
• Primary keyword near beginning
• Natural language, human readable
• No keyword stuffing, no clickbait
• Match page type and search intent
• 50–60 characters preferred, under 65 maximum
• Avoid repetitive structures
• Do NOT make every title a Complete/Ultimate Guide

## META RULES
One unique meta description per title.
• 145–160 characters
• Include primary keyword + one supporting keyword
• Clear value proposition
• CTA when appropriate
• Natural language
• Do NOT repeat the title
• Avoid filler

## CTR
Use Trusted / Professional / Reliable / Expert / Affordable / Fast / Comprehensive / Free Quote / 2026 only when relevant.
Avoid excessive marketing language.

## QUALITY
Reject outputs that use generic templates, misrepresent the page, change the topic, stuff keywords, repeat phrases, exceed length, or sound AI-generated.
Every title must be specific to the detected page type.
If the same title could work for thousands of unrelated pages, reject it.

Return JSON only. No markdown fences. No commentary.

Schema:
{
  "understanding": {
    "primary_topic": "string",
    "secondary_topics": ["string"],
    "primary_keyword": "string",
    "supporting_keywords": ["string"],
    "named_entities": ["string"],
    "industry": "string",
    "target_audience": "string",
    "search_intent": "Informational|Commercial|Transactional|Navigational|Mixed",
    "intent_confidence": 0,
    "confidence": 0
  },
  "page_type": "Homepage|Service Page|Local Service Page|Landing Page|Product Page|Category Page|Blog|Educational Guide|Tutorial|Case Study|Brand Story|News|Comparison|Review|Documentation|FAQ",
  "commercial_intent": {
    "label": "Informational|Commercial|Transactional|Navigational|Mixed",
    "confidence": 0
  },
  "serp_analysis": {
    "dominant_style": "string",
    "top_patterns": ["up to 3"]
  },
  "title_strategy": "Brand|Service|Educational|Question|Benefit|Comparison|Listicle|Local SEO|Problem/Solution|Authority",
  "keyword_analysis": {
    "primary_keyword": "string",
    "supporting_keywords": ["string"],
    "notes": "string"
  },
  "variations": [
    {
      "title": "string",
      "meta_description": "string",
      "seo_score": 0,
      "ctr_score": 0,
      "overall_score": 0,
      "reasoning": "1 sentence why this fits THIS page type"
    }
  ],
  "validation": {
    "rejected_examples": [{"title": "string", "reason": "string"}],
    "passed": true
  }
}
"""


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _extract_json(raw: str) -> dict[str, Any]:
  text = (raw or "").strip()
  if not text:
    raise ValueError("empty title-meta strategist response")
  fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
  if fence:
    text = fence.group(1).strip()
  start = text.find("{")
  end = text.rfind("}")
  if start < 0 or end <= start:
    raise ValueError("no JSON object in title-meta strategist response")
  return json.loads(text[start : end + 1])


def _clip_title(title: str) -> str:
  t = _clean(title)
  if len(t) <= TITLE_HARD_MAX:
    return t
  cut = t[: TITLE_HARD_MAX - 1].rsplit(" ", 1)[0]
  return cut or t[:TITLE_HARD_MAX]


def _clip_meta(meta: str) -> str:
  m = _clean(meta)
  if len(m) > META_MAX:
    m = m[: META_MAX - 1].rsplit(" ", 1)[0]
  if len(m) < META_MIN and m:
    # Soft pad only with natural clause if slightly short — do not invent fluff slogans
    pad = " Learn more."
    while len(m) < META_MIN and len(m) + len(pad) <= META_MAX:
      m = (m.rstrip(".") + pad).strip()
      break
  return m[:META_MAX]


def _page_type_allows_guide(page_type: str) -> bool:
  return (page_type or "").strip().lower() in _GUIDE_OK_TYPES


def _is_banned_title(title: str, page_type: str) -> bool:
  low = title.lower()
  if _page_type_allows_guide(page_type):
    # Still reject ultra-generic complete/ultimate guide spam on blogs if overused patterns
    if "everything you need to know" in low:
      return True
    return False
  for frag in _BANNED_TITLE_FRAGMENTS:
    if frag in low:
      return True
  if re.search(r"\b(tips|strategies)\b", low) and "service" in (page_type or "").lower():
    return True
  return False


def _is_generic_title(title: str, topic: str) -> bool:
  t = title.lower()
  topic_tokens = [w for w in re.findall(r"[a-z0-9]+", (topic or "").lower()) if len(w) > 2]
  if not topic_tokens:
    return False
  hits = sum(1 for w in topic_tokens if w in t)
  # Title shares almost nothing with the topic → too generic
  return hits == 0 and len(topic_tokens) >= 2


def build_title_meta_strategist_prompt(
  *,
  topic: str,
  variations: int,
  category: str,
  tone: str,
  language: str,
) -> str:
  return (
    f"Language: {language or 'English'}\n"
    f"Tone hint: {tone or 'professional'}\n"
    f"Category hint: {category or 'blog_article'}\n"
    f"Requested variations: {variations} (generate between 5 and {variations} unique title+meta pairs)\n"
    f"Input topic / page brief:\n{topic}\n\n"
    "Detect page type from this input. Generate page-type-specific titles and metas. "
    "Return JSON only."
  )


def strategist_payload_to_result(
  data: dict[str, Any],
  *,
  topic: str,
  variations: int,
  category: str,
  tone: str,
  language: str,
  elapsed_ms: float,
  backend: str,
) -> dict[str, Any]:
  understanding = data.get("understanding") if isinstance(data.get("understanding"), dict) else {}
  page_type = _clean(str(data.get("page_type") or "Blog"))
  commercial = data.get("commercial_intent") if isinstance(data.get("commercial_intent"), dict) else {}
  serp = data.get("serp_analysis") if isinstance(data.get("serp_analysis"), dict) else {}
  kw_analysis = data.get("keyword_analysis") if isinstance(data.get("keyword_analysis"), dict) else {}
  strategy = _clean(str(data.get("title_strategy") or "Benefit"))
  validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}

  primary_kw = _clean(
    str(
      kw_analysis.get("primary_keyword")
      or understanding.get("primary_keyword")
      or understanding.get("primary_topic")
      or topic
    )
  )

  items: list[dict[str, Any]] = []
  seen: set[str] = set()
  rejected: list[dict[str, str]] = []

  for raw in data.get("variations") or []:
    if not isinstance(raw, dict):
      continue
    title = _clip_title(str(raw.get("title") or ""))
    meta = _clip_meta(str(raw.get("meta_description") or raw.get("meta") or ""))
    if not title or not meta:
      continue
    if _is_banned_title(title, page_type):
      rejected.append({"title": title, "reason": f"banned template phrase for page type {page_type}"})
      continue
    if _is_generic_title(title, topic):
      rejected.append({"title": title, "reason": "too generic / not specific to topic"})
      continue
    if len(title) > TITLE_HARD_MAX:
      rejected.append({"title": title, "reason": "title too long"})
      continue
    key = title.lower()
    if key in seen:
      rejected.append({"title": title, "reason": "duplicate"})
      continue
    seen.add(key)

    q = engine.quality_variation(title, meta, topic)
    try:
      seo_score = int(raw.get("seo_score") or q.get("quality_score") or 0)
    except (TypeError, ValueError):
      seo_score = int(q.get("quality_score") or 0)
    try:
      ctr_score = int(raw.get("ctr_score") or max(50, seo_score - 5))
    except (TypeError, ValueError):
      ctr_score = max(50, seo_score - 5)
    try:
      overall = int(raw.get("overall_score") or round((seo_score + ctr_score) / 2))
    except (TypeError, ValueError):
      overall = round((seo_score + ctr_score) / 2)

    # Prefer titles in 50–60 band
    if 50 <= len(title) <= TITLE_SOFT_MAX:
      overall = min(100, overall + 3)
    elif len(title) > TITLE_SOFT_MAX:
      overall = max(0, overall - 4)

    issues = list(q.get("issues") or [])
    if META_MIN <= len(meta) <= META_MAX:
      pass
    elif len(meta) < engine.META_MIN:
      issues.append("meta_short")
    reasoning = _clean(str(raw.get("reasoning") or ""))

    items.append({
      "title": title,
      "title_length": len(title),
      "meta_description": meta,
      "meta_length": len(meta),
      "angle": strategy.lower().replace(" ", "_") or "strategist",
      "quality_score": int(q.get("quality_score") or overall),
      "seo_score": max(0, min(100, seo_score)),
      "ctr_score": max(0, min(100, ctr_score)),
      "overall_score": max(0, min(100, overall)),
      "seo_ready": bool(q.get("seo_ready")) or overall >= 70,
      "issues": issues,
      "reasoning": reasoning,
    })
    if len(items) >= variations:
      break

  # Sort best first
  items.sort(key=lambda x: (-int(x.get("overall_score") or 0), x.get("title_length", 99)))

  if len(items) < 5:
    raise ValueError(
      f"title-meta strategist returned too few valid variations ({len(items)}); "
      f"rejected={len(rejected)}"
    )

  avg_quality = int(round(sum(v.get("quality_score", 0) for v in items) / max(len(items), 1)))
  avg_seo = int(round(sum(v.get("seo_score", 0) for v in items) / max(len(items), 1)))
  avg_ctr = int(round(sum(v.get("ctr_score", 0) for v in items) / max(len(items), 1)))

  for r in validation.get("rejected_examples") or []:
    if isinstance(r, dict) and r.get("title"):
      rejected.append({"title": str(r["title"]), "reason": str(r.get("reason") or "model rejected")})

  flow = [
    "input",
    "understanding",
    "page_type_detection",
    "commercial_intent",
    "serp_analysis",
    "title_strategy",
    "title_generation",
    "meta_generation",
    "ctr_optimization",
    "quality_validation",
  ]

  return {
    "topic": topic,
    "category": category,
    "language": language or "en",
    "tone": tone,
    "variations": items[:variations],
    "variation_count": min(len(items), variations),
    "title_limit": TITLE_SOFT_MAX,
    "meta_min": META_MIN,
    "meta_max": META_MAX,
    "quality": {
      "average_score": avg_quality,
      "seo_ready": avg_quality >= 75,
      "all_ready": all(v.get("seo_ready") for v in items[:variations]),
      "average_seo_score": avg_seo,
      "average_ctr_score": avg_ctr,
    },
    "ai": {
      "enabled": True,
      "model_used": True,
      "backend": backend,
      "hosted": True,
      "mode": "title_meta_strategist_v1",
    },
    "generator_version": STRATEGIST_VERSION,
    "variation_seed": None,
    "policy": {"safe_mode": False, "page_type": page_type},
    "page_type": page_type,
    "title_strategy": strategy,
    "understanding": understanding,
    "commercial_intent": commercial,
    "serp_analysis": serp,
    "keyword_analysis": {
      "primary_keyword": primary_kw,
      "supporting_keywords": [
        _clean(str(x))
        for x in (kw_analysis.get("supporting_keywords") or understanding.get("supporting_keywords") or [])
        if _clean(str(x))
      ][:8],
      "notes": _clean(str(kw_analysis.get("notes") or "")),
    },
    "architecture": {
      "flow": flow,
      "stages": {
        "page_type": page_type,
        "title_strategy": strategy,
        "understanding": understanding,
        "serp_analysis": serp,
        "validation": {"rejected": rejected[:12], "kept": len(items)},
      },
    },
    "pipeline": {
      "page_type": page_type,
      "search_intent": commercial.get("label") or understanding.get("search_intent"),
      "title_strategy": strategy,
      "serp_patterns": serp.get("top_patterns") or [],
      "primary_keyword": primary_kw,
    },
    "rag": {"enabled": False, "sources_used": []},
    "elapsed_ms": round(elapsed_ms, 1),
    "unlimited_outputs": True,
    "strategist": data,
  }


async def generate_with_title_meta_strategist(
  provider: ModelProvider,
  *,
  topic: str,
  variations: int = 10,
  category: str = "blog_article",
  tone: str = "professional",
  language: str = "English",
) -> dict[str, Any]:
  topic = _clean(topic)
  if not topic:
    raise ValueError("topic is required")
  n = max(5, min(50, int(variations or 10)))

  t0 = time.perf_counter()
  user_prompt = build_title_meta_strategist_prompt(
    topic=topic,
    variations=n,
    category=category,
    tone=tone,
    language=language,
  )
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=_SYSTEM,
    use_rag=False,
    skip_intent=True,
    skip_kb_direct_match=True,
    max_tokens=4500,
    temperature=0.35,
  )
  data = _extract_json(raw)
  backend = getattr(provider, "last_backend", None) or getattr(provider, "model_id", lambda: "hosted")()
  if callable(backend):
    try:
      backend = backend()
    except TypeError:
      pass
  elapsed = (time.perf_counter() - t0) * 1000
  return strategist_payload_to_result(
    data,
    topic=topic,
    variations=n,
    category=category,
    tone=tone,
    language=language or "en",
    elapsed_ms=elapsed,
    backend=str(backend) if backend else "hosted",
  )

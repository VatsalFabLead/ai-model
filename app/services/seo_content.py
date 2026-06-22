"""SEO Content Generator — advanced, multilingual, worldwide.

Single generate() entry with optional web keyword discovery, category templates,
training knowledge fallbacks, and custom model enhancement. No GPT/Claude/Gemini.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.engine import seo_content_engine
from app.engine.keyword_discovery import discover_keywords
from app.services.provider_base import ModelProvider

_GENERIC_HEADINGS = {
  "introduction", "intro", "overview", "summary", "contents", "table of contents",
  "getting started", "background", "conclusion", "about",
}


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
  t = text.strip()
  if not t.startswith("{"):
    return None
  try:
    obj = json.loads(t)
    return obj if isinstance(obj, dict) else None
  except Exception:
    return None


def _is_valid_body(body: str, min_words: int = 80) -> bool:
  if not body or _count_words(body) < min_words:
    return False
  if body.count("###") > 8 or "Training Knowledge" in body:
    return False
  return "##" in body or _count_words(body) >= min_words + 40


def _parse_seo(raw: str, topic: str) -> tuple[str, str, str]:
  text = (raw or "").strip()
  obj = _try_json(text)
  if obj:
    title = (obj.get("title") or "").strip()
    meta = (obj.get("meta_description") or obj.get("meta") or obj.get("description") or "").strip()
    body = (obj.get("content") or obj.get("body") or obj.get("article") or "").strip()
    if body:
      title = title or topic.strip().title()[:70]
      meta = meta or body
      return title, _trim_meta(meta), body

  title = None
  meta = None
  m = re.search(r"^\s*TITLE\s*[:\-]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
  if m:
    title = m.group(1).strip().strip('"\u201c\u201d')
  m2 = re.search(
    r"^\s*(?:META(?:\s*DESCRIPTION)?|DESCRIPTION)\s*[:\-]\s*(.+)$",
    text, re.IGNORECASE | re.MULTILINE,
  )
  if m2:
    meta = m2.group(1).strip().strip('"\u201c\u201d')

  body = re.sub(r"^\s*TITLE\s*[:\-].+$", "", text, flags=re.IGNORECASE | re.MULTILINE)
  body = re.sub(
    r"^\s*(?:META(?:\s*DESCRIPTION)?|DESCRIPTION)\s*[:\-].+$", "",
    body, flags=re.IGNORECASE | re.MULTILINE,
  )
  body = re.sub(r"^\s*-{3,}\s*$", "", body, flags=re.MULTILINE).strip()

  if not title:
    mh = re.search(r"^#+\s*(.+)$", body, re.MULTILINE)
    cand = re.sub(r"[*_`]", "", mh.group(1)).strip() if mh else ""
    if cand and cand.lower() not in _GENERIC_HEADINGS and len(cand.split()) >= 3:
      title = cand
    else:
      title = topic.strip().title()[:70]

  if not meta:
    para = None
    for block in re.split(r"\n\s*\n", body):
      b = block.strip()
      if b and not b.startswith("#"):
        para = b
        break
    meta = para or topic

  return title, _trim_meta(meta), body


def _fallback_article(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  tone: str,
  target_words: int,
  audience: str | None,
  language: str | None,
) -> tuple[str, str, str]:
  """Deterministic aesthetic article when the model output is weak."""
  primary = keywords[0] if keywords else topic.strip()
  year = "2026"
  title = f"{primary.title()}: The Complete {year} Guide"
  if category == "how_to_guide":
    title = f"How to Master {primary.title()} ({year} Step-by-Step Guide)"
  elif category == "listicle":
    title = f"10 Proven {primary.title()} Strategies That Work in {year}"
  elif category == "local_seo" and audience:
    title = f"Best {primary.title()} in {audience} — Local Expert Guide"

  meta = _trim_meta(
    f"Discover practical {primary} strategies for worldwide audiences. "
    f"Actionable tips, expert insights, and proven steps to get real results in {year}."
  )

  kw_extra = ", ".join(keywords[1:4]) if len(keywords) > 1 else primary
  audience_line = f" Whether you are in {audience} or anywhere globally," if audience else " Worldwide,"

  sections: list[str] = []
  if category == "how_to_guide":
    sections = [
      "## What You Need Before You Start",
      f"Before diving into **{primary}**, gather your basics: clear goals, the right tools, and realistic time. This foundation saves hours later.",
      "## Step 1: Understand the Fundamentals",
      f"Start with core concepts of {primary}. Learn how {kw_extra} connect to your main objective.",
      "## Step 2: Apply Proven Techniques",
      "Follow industry best practices: measure results, iterate quickly, and document what works for your audience.",
      "## Step 3: Optimize and Scale",
      f"Refine your approach to {primary} using data. Small improvements compound into significant growth over time.",
      "## Common Mistakes to Avoid",
      "- Skipping research on audience intent\n- Keyword stuffing instead of helpful content\n- Ignoring mobile and page speed",
    ]
  elif category == "listicle":
    sections = [
      f"## 1. Start With a Clear {primary.title()} Strategy",
      "Define goals, audience, and success metrics before creating content or campaigns.",
      f"## 2. Use the Right Tools for {primary.title()}",
      f"Pick reliable platforms that support {kw_extra} workflows without unnecessary complexity.",
      "## 3. Create High-Quality, Original Content",
      "Publish helpful articles, guides, and updates that answer real user questions.",
      "## 4. Measure and Improve Continuously",
      "Track rankings, traffic, and conversions — then double down on what performs.",
      "## 5. Stay Updated With Trends",
      f"The {primary} landscape evolves fast. Follow trusted sources and adapt quarterly.",
    ]
  else:
    sections = [
      f"## Why {primary.title()} Matters in {year}",
      f"{audience_line} businesses and creators who invest in {primary} gain visibility, trust, and sustainable growth.",
      f"## Key Benefits of Strong {primary.title()}",
      f"- Better search visibility for {kw_extra}\n- Higher engagement from target readers\n- Long-term compounding traffic",
      "## Best Practices That Work Worldwide",
      "Focus on user intent, original research, clear structure, and consistent publishing. Avoid shortcuts that trigger penalties.",
      f"## How to Get Started With {primary.title()}",
      "Audit your current content, identify gaps versus competitors, and publish one high-quality piece per week.",
      "## Conclusion",
      f"Mastering **{primary}** takes patience and consistency. Apply these steps, track results, and refine your strategy every month.",
    ]

  body = "\n\n".join(sections)
  # Pad toward target length with an FAQ block
  body += (
    f"\n\n## Frequently Asked Questions\n\n"
    f"### What is {primary}?\n"
    f"{primary.title()} is a proven approach used by professionals worldwide to improve visibility and results.\n\n"
    f"### How long does {primary} take to show results?\n"
    "Most strategies show meaningful progress within 8–12 weeks with consistent effort.\n\n"
    f"### Who should focus on {primary}?\n"
    f"Marketers, business owners, and creators who want sustainable growth in search and social channels."
  )
  return title, meta, body


async def generate(
  provider: ModelProvider,
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
) -> dict[str, Any]:
  """Single SEO content generator — keywords, AI article, quality score."""
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
  lang_line = f" Write the entire article in {language} (language code {lang_code})." if language else ""
  structure = seo_content_engine.category_structure_hint(cat)

  ai_used = False
  title, meta, body = "", "", ""

  if use_ai:
    tone_guide = seo_content_engine.tone_hint(tone_str)
    system_prompt = (
      f"You are an expert worldwide SEO content writer. Write in a {tone_str} tone "
      f"({tone_guide}) Create an original, SEO-optimized {cat.replace('_', ' ')} "
      f"of about {target} words.{lang_line}{audience_line} "
      f"Structure: {structure} "
      "Rules: primary keyword in title and first paragraph; use ## and ### markdown headings; "
      "natural keyword integration; intro + conclusion with CTA. "
      "Respond EXACTLY in this format:\n"
      "TITLE: <SEO title under 60 chars>\n"
      "META: <meta description under 160 chars>\n"
      "---\n"
      "<full article markdown body>"
    )
    user_prompt = (
      f"Topic: {topic}\nPrimary keyword: {primary}\nKeywords: {kw_line}\nCategory: {cat}"
    )
    try:
      max_tokens = min(1100, int(target * 1.6) + 80)
      raw = await provider.chat(
        [{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        use_rag=False,
        skip_intent=True,
        max_tokens=max_tokens,
        temperature=0.65,
      )
      title, meta, body = _parse_seo(raw, topic)
      body = _clean_body(body)
      if not _is_valid_body(body, min_words=max(80, target // 4)):
        raise ValueError("weak ai body")
      ai_used = True
    except Exception:
      title, meta, body = _fallback_article(
        topic, kws, category=cat, tone=tone_str, target_words=target,
        audience=audience, language=language,
      )
  else:
    title, meta, body = _fallback_article(
      topic, kws, category=cat, tone=tone_str, target_words=target,
      audience=audience, language=language,
    )

  quality = seo_content_engine.quality_report(title, meta, body, kws)
  return {
    "topic": topic,
    "category": cat,
    "language": lang_code,
    "tone": tone_str,
    "title": title,
    "meta_description": meta,
    "slug": _slugify(title),
    "keywords": kws,
    "content": body,
    "word_count": _count_words(body),
    "quality": quality,
    "discovery": discovery_meta,
    "ai": {"enabled": use_ai, "model_used": ai_used},
  }


# Backward-compatible alias
async def generate_seo_content(provider: ModelProvider, **kwargs) -> dict:
  return await generate(provider, **kwargs)

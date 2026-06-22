"""SEO Title & Meta Description Generator — advanced, multilingual.

Single generate() endpoint with categories, 4 tones, quality scores,
training knowledge, and custom model + template fallback.
No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import json
import re

from app.engine import title_meta_engine as engine
from app.services.provider_base import ModelProvider

TITLE_MAX = engine.TITLE_MAX
META_MIN = engine.META_MIN
META_MAX = engine.META_MAX

_POWER_WORDS = (
  "Ultimate", "Essential", "Proven", "Complete", "Expert", "Smart",
  "Powerful", "Simple", "Best", "Top",
)


def supported_categories() -> list[dict[str, str]]:
  return engine.supported_categories()


def supported_tones() -> list[dict[str, str]]:
  return engine.supported_tones()


def supported_languages() -> list[dict[str, str]]:
  return engine.supported_languages()


def _trim_title(title: str, limit: int = TITLE_MAX) -> str:
  title = re.sub(r"\*+", "", (title or "").strip()).strip('"\u201c\u201d\'')
  title = re.sub(r"\s+", " ", title)
  if len(title) <= limit:
    return title
  cut = title[:limit]
  space = cut.rfind(" ")
  if space > limit * 0.55:
    cut = cut[:space]
  return cut.rstrip(" -:|,") + ("…" if len(title) > limit else "")


def _trim_meta(meta: str, min_len: int = META_MIN, max_len: int = META_MAX) -> str:
  meta = re.sub(r"\*+", "", (meta or "").strip()).strip('"\u201c\u201d\'')
  meta = re.sub(r"^\s*[\-\*]\s*", "", meta)
  meta = re.sub(r"Hook/Angle.*?:", "", meta, flags=re.IGNORECASE)
  meta = re.sub(r"\s+", " ", meta)
  if len(meta) > max_len:
    cut = meta[: max_len - 3]
    space = cut.rfind(" ")
    if space > max_len * 0.6:
      cut = cut[:space]
    meta = cut.rstrip(" ,.;:") + "..."
  if len(meta) < min_len and meta:
    meta = meta + " Learn more and get started today."
    if len(meta) > max_len:
      meta = meta[: max_len - 3].rsplit(" ", 1)[0] + "..."
  return meta


def _topic_clean(topic: str) -> str:
  return re.sub(r"\s+", " ", topic.strip())


def _topic_title_case(topic: str) -> str:
  t = _topic_clean(topic)
  return t[0].upper() + t[1:] if t else "Your Topic"


def _variation_item(title: str, meta: str, topic: str, angle: str = "") -> dict:
  t = _trim_title(title)
  m = _trim_meta(meta)
  q = engine.quality_variation(t, m, topic)
  return {
    "title": t,
    "title_length": len(t),
    "meta_description": m,
    "meta_length": len(m),
    "angle": angle,
    "quality_score": q["quality_score"],
    "seo_ready": q["seo_ready"],
    "issues": q["issues"],
  }


def _template_variations(topic: str, count: int, tone: str, category: str) -> list[dict]:
  base = _topic_title_case(topic)
  low = base.lower()
  year = "2026"

  if category == "local_business":
    templates = [
      (f"Best {base} Services Near You ({year})", f"Looking for trusted {low}? Discover top-rated local experts, fast response, and proven results. Contact us today.", "local"),
      (f"{base} in Your Area — Trusted Local Experts", f"Get reliable {low} from experienced professionals in your community. Free quotes and friendly service.", "trust"),
    ]
  elif category == "product_page" or category == "ecommerce":
    templates = [
      (f"Shop {base} — Best Prices & Fast Delivery", f"Buy {low} online with confidence. Top quality, great reviews, and secure checkout. Order today.", "shop"),
      (f"{base}: Features, Reviews & Buy Online", f"Discover why customers love our {low}. Compare specs, read reviews, and shop with free returns.", "product"),
    ]
  elif category == "how_to":
    templates = [
      (f"How to Master {base} ({year} Guide)", f"Learn {low} step by step with this practical {year} guide. Simple tips that work for beginners and pros.", "how-to"),
      (f"{base}: Step-by-Step Tutorial for Beginners", f"New to {low}? Follow our easy tutorial and start seeing results fast. Expert tips included.", "beginner"),
    ]
  elif tone == "casual":
    templates = [
      (f"{base}? Here's What Actually Works", f"Skip the fluff — here's the real deal on {low}. Quick tips you can use today.", "casual"),
      (f"Your No-Stress Guide to {base}", f"Making {low} easy and fun. Practical advice without the jargon. Dive in!", "relaxed"),
    ]
  elif tone == "friendly":
    templates = [
      (f"Welcome to Your {base} Guide", f"We're here to help you succeed with {low}. Friendly tips, clear steps, and support every step of the way.", "friendly"),
      (f"Let's Talk About {base} — Simple Tips Inside", f"Everything you wanted to know about {low}, explained in plain language. Start your journey today.", "warm"),
    ]
  elif tone == "formal":
    templates = [
      (f"{base}: A Comprehensive Overview ({year})", f"An authoritative examination of {low} principles, applications, and best practices for professional audiences.", "formal"),
      (f"Analysis of {base} — Key Insights & Recommendations", f"This overview presents essential findings on {low} for informed decision-making.", "academic"),
    ]
  else:
    templates = [
      (f"{base}: The Complete {year} Guide", f"Discover proven strategies for {low}. Expert tips, actionable steps, and real results in {year}.", "guide"),
      (f"How to Master {base} ({year} Tips)", f"Learn how to excel at {low} with this practical guide. Techniques used by professionals worldwide.", "how-to"),
      (f"{_POWER_WORDS[0]} {base} Strategies That Work", f"Boost your results with proven {low} strategies. Expert advice — start improving today.", "power"),
      (f"Top 10 {base} Best Practices", f"Explore the top 10 best practices for {low}. Increase success with these expert tips.", "listicle"),
      (f"{base} — Expert Guide for Beginners", f"New to {low}? This guide covers essentials, mistakes to avoid, and quick wins.", "beginner"),
    ]

  out: list[dict] = []
  for title, meta, angle in templates[:count]:
    out.append(_variation_item(title, meta, topic, angle))
  return out


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
              variations.append(_variation_item(title, meta, topic))
        if variations:
          return variations
    except Exception:
      pass

  blocks = re.split(r"(?:\n\s*---\s*\n|\n\s*Variation\s+\d+\s*[:\-]?\s*\n)", text, flags=re.IGNORECASE)
  if len(blocks) <= 1:
    blocks = re.split(r"\n\s*\n\s*(?=TITLE\s*[:\-])", text, flags=re.IGNORECASE)

  for block in blocks:
    block = block.strip()
    if not block:
      continue
    title = meta = None
    m = re.search(r"TITLE\s*[:\-]\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
    if m:
      title = m.group(1).strip()
    m2 = re.search(
      r"(?:META(?:\s*DESCRIPTION)?|DESCRIPTION)\s*[:\-]\s*(.+?)(?:\n\s*\n|$)",
      block, re.IGNORECASE | re.DOTALL,
    )
    if m2:
      meta = m2.group(1).strip().split("\n")[0].strip()
    if title and meta:
      item = _variation_item(title, meta, topic)
      if _is_valid_item(item["title"], item["meta_description"]):
        variations.append(item)
    if len(variations) >= count:
      break
  return variations


def _is_weak_response(text: str) -> bool:
  low = (text or "").lower()
  return (
    "don't have a confident answer" in low
    or len(text.strip()) < 40
    or not re.search(r"title\s*[:\-]", text, re.IGNORECASE)
  )


def _is_valid_item(title: str, meta: str) -> bool:
  if not title or not meta:
    return False
  if "**" in title or "Hook/Angle" in meta:
    return False
  if title.startswith("-") or meta.startswith("-"):
    return False
  if len(title) < 10 or len(meta) < 50:
    return False
  return True


async def generate(
  provider: ModelProvider,
  *,
  topic: str,
  variations: int = 3,
  tone: str | None = None,
  language: str | None = None,
  category: str | None = None,
  use_ai: bool = True,
) -> dict:
  topic = _topic_clean(topic)
  if not topic:
    raise ValueError("topic is required")

  n = max(1, min(5, variations))
  cat = engine.normalize_category(category)
  tone_str = engine.normalize_tone(tone, cat)
  lang_code = engine.bcp47(language)
  ai_used = False
  items: list[dict] = []

  if use_ai:
    lang_line = f" Write all output in {language}." if language else ""
    tone_guide = engine.tone_hint(tone_str)
    system_prompt = (
      f"You are an expert worldwide SEO copywriter. Create {n} distinct title + meta "
      f"pairs in a {tone_str} tone ({tone_guide}). Category: {cat.replace('_', ' ')}.{lang_line}\n"
      f"Rules:\n"
      f"- Title: max {TITLE_MAX} chars; keyword near start\n"
      f"- Meta: {META_MIN}-{META_MAX} chars; benefit + CTA\n"
      "- Each variation must use a different hook/angle\n"
      "- No markdown, no explanations\n"
      "Format EXACTLY:\n"
      "TITLE: <title>\nMETA: <meta>\n---\n"
      f"(repeat {n} times)"
    )
    try:
      raw = await provider.chat(
        [{"role": "user", "content": f"Topic / keyword: {topic}"}],
        system_prompt=system_prompt,
        use_rag=False,
        skip_intent=True,
        max_tokens=min(650, 80 + n * 130),
        temperature=0.7,
      )
      if _is_weak_response(raw):
        raise ValueError("weak ai")
      items = _parse_variations(raw, topic, n)
      if len(items) < 1:
        raise ValueError("no valid variations parsed")
      ai_used = True
    except Exception:
      items = _template_variations(topic, n, tone_str, cat)
  else:
    items = _template_variations(topic, n, tone_str, cat)

  if len(items) < n:
    existing = {v["title"].lower() for v in items}
    for tpl in _template_variations(topic, n, tone_str, cat):
      if tpl["title"].lower() not in existing:
        items.append(tpl)
        existing.add(tpl["title"].lower())
      if len(items) >= n:
        break

  items = items[:n]
  avg_quality = int(round(sum(v["quality_score"] for v in items) / max(len(items), 1)))

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
      "all_ready": all(v["seo_ready"] for v in items),
    },
    "ai": {"enabled": use_ai, "model_used": ai_used},
  }


async def generate_title_meta(provider: ModelProvider, **kwargs) -> dict:
  return await generate(provider, **kwargs)

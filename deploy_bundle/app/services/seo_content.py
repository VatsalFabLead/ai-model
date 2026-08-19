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
from app.engine.seo_content_rag_pipeline import (
  GENERATOR_VERSION,
  ARCHITECTURE_FLOW,
  run_seo_content_pipeline,
)
from app.services.provider_base import ModelProvider

_GENERIC_HEADINGS = {
  "introduction", "intro", "overview", "summary", "contents", "table of contents",
  "getting started", "background", "conclusion", "about",
}

_AI_TIMEOUT_SEC = 75.0

_INSTRUCTION_PREFIX_RE = re.compile(
  r"^\s*(?:please\s+)?"
  r"(?:generate|write|create|make|draft|produce)\s+"
  r"(?:(?:an?\s+|some\s+)?(?:seo\s+)?(?:content|article|blog(?:\s*post)?|post|guide)\s+)?"
  r"(?:for|about|on|regarding)\s*[:\-]?\s*",
  re.IGNORECASE,
)

_BOILERPLATE_MARKERS = (
  "when exploring",
  "plays a key role in successful",
  "apply industry best practices and measure results",
  "focus on verified information about",
  "avoid unrelated sources",
  "search-aligned guidance",
  "learn how ",
  "applies to ",
  "generate seo content for",
)


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
    parts = re.split(r"[,;\n|]+", keywords)
  else:
    parts = [str(k) for k in keywords]
  seen: set[str] = set()
  out: list[str] = []
  for p in parts:
    p = sanitize_topic(p)
    if not p or _looks_like_instruction(p):
      continue
    if p.lower() not in seen:
      seen.add(p.lower())
      out.append(p)
  return out


def sanitize_topic(topic: str) -> str:
  """Strip prompt wrappers like 'Generate SEO content for Coffee' → 'Coffee'."""
  t = re.sub(r"\s+", " ", (topic or "").strip())
  if not t:
    return ""
  prev = None
  while prev != t:
    prev = t
    t = _INSTRUCTION_PREFIX_RE.sub("", t).strip(" -:|,.")
  # Drop trailing brief junk: "Coffee. Keywords: best practices..."
  t = re.split(r"\b(?:keywords?|tone|word\s*count|audience)\s*[:\-]", t, maxsplit=1, flags=re.I)[0]
  return t.strip(" -:|,.")[:300]


def _looks_like_instruction(text: str) -> bool:
  low = (text or "").lower()
  return bool(
    re.search(r"\b(generate|write|create)\s+(seo\s+)?(content|article|blog)\b", low)
    or low.startswith("generate seo")
    or low.startswith("write an article")
  )


def _is_boilerplate_article(article: str) -> bool:
  low = (article or "").lower()
  if not low.strip():
    return True
  hits = sum(1 for m in _BOILERPLATE_MARKERS if m in low)
  if hits >= 2:
    return True
  # Heavy repetition of the same short sentence pattern
  sentences = [s.strip() for s in re.split(r"[.!?]\s+", low) if len(s.strip()) > 40]
  if len(sentences) >= 4:
    unique = len(set(sentences))
    if unique / len(sentences) < 0.45:
      return True
  return False


def _strip_boilerplate_lines(article: str) -> str:
  keep: list[str] = []
  for ln in (article or "").splitlines():
    low = ln.lower().strip()
    if any(m in low for m in _BOILERPLATE_MARKERS) and not low.startswith("#"):
      continue
    keep.append(ln)
  return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def _slugify(text: str, max_len: int = 60) -> str:
  text = (text or "").lower().strip()
  text = re.sub(r"[^a-z0-9\s-]", "", text)
  text = re.sub(r"[\s_-]+", "-", text).strip("-")
  if len(text) > max_len:
    text = text[:max_len].rsplit("-", 1)[0]
  return text or "untitled"


def _suggest_tags(
  *,
  topic: str,
  title: str,
  primary: str,
  secondary: list[str],
  outline: list[dict[str, str]] | list[str],
  category: str,
  ai_tags: list[str] | None = None,
  max_tags: int = 12,
) -> list[str]:
  """Build SEO-friendly suggested tags from keywords, outline, and optional AI tags."""
  tags: list[str] = []
  seen: set[str] = set()

  def _add(raw: str) -> None:
    t = re.sub(r"\s+", " ", (raw or "").strip(" #,;|/"))
    if not t or len(t) < 2 or len(t) > 48:
      return
    # Drop instruction-like / filler tags
    low = t.lower()
    if low in seen:
      return
    if re.search(r"\b(generate|write|create)\s+(seo\s+)?(content|article)\b", low):
      return
    if low in ("introduction", "conclusion", "overview", "summary", "faq", "faqs"):
      return
    seen.add(low)
    tags.append(t)

  for item in ai_tags or []:
    _add(str(item))
  _add(primary)
  _add(topic)
  for kw in secondary:
    _add(kw)
  # Short phrases from H2 headings
  for item in outline or []:
    text = item.get("text") if isinstance(item, dict) else str(item)
    level = (item.get("level") if isinstance(item, dict) else "h2") or "h2"
    if str(level).lower() == "h2" and text:
      _add(text)
  # Category-style tag
  if category:
    _add(category.replace("_", " "))
  # Title words as last resort for coverage
  if title and len(tags) < 5:
    for part in re.split(r"[:\-–—|]", title):
      _add(part.strip())

  return tags[:max_tags]


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


def _repair_json(raw: str) -> str:
  """Clean up common LLM JSON syntax anomalies (unescaped quotes, markdown wrappers, trailing commas)."""
  text = (raw or "").strip()
  text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
  text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
  text = re.sub(r",\s*([\}\]])", r"\1", text)
  return text.strip()


def _try_json(text: str) -> dict | None:
  t = _repair_json(text)
  if not t:
    return None
  if not t.startswith("{"):
    m = re.search(r"\{[\s\S]*\}", t)
    t = m.group(0) if m else t
  try:
    obj = json.loads(t)
    return obj if isinstance(obj, dict) else None
  except Exception:
    try:
      t_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', t)
      obj = json.loads(t_clean)
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
  if obj:
    meta_obj = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    content_obj = obj.get("content") if isinstance(obj.get("content"), dict) else {}

    title = (meta_obj.get("title") or obj.get("title") or "").strip()
    meta = (
      meta_obj.get("meta_description") or obj.get("meta_description") or obj.get("meta") or ""
    ).strip()
    article = (
      content_obj.get("article") or obj.get("article") or obj.get("body") or ""
    ).strip()
    if isinstance(article, dict):
      article = str(article.get("article") or article.get("body") or "").strip()
    tone = (content_obj.get("tone") or obj.get("tone") or "").strip()

    outline = _coerce_outline_struct(obj.get("outline"))
    faqs = _coerce_faqs(obj.get("faqs"))
    keywords = _coerce_keywords_struct(obj.get("keywords"), [])

    if article and _count_words(article) >= 80 and not _is_boilerplate_article(article):
      if not title:
        title = topic.strip().title()[:70]
      if not meta:
        meta = _trim_meta(re.sub(r"[#*_>`]", "", article)[:220])
      article = seo_content_engine.strip_faq_section(_clean_body(_strip_boilerplate_lines(article)))
      if not outline:
        extracted = seo_content_engine.extract_outline_from_body(article)
        outline = [{"level": "h2", "text": t} for t in extracted]
      if not faqs:
        faqs = seo_content_engine.extract_faqs_from_body(obj.get("content") or article)
      if not keywords.get("primary"):
        keywords = {"primary": topic, "secondary": []}
      slug_raw = (
        obj.get("slug")
        or meta_obj.get("slug")
        or ""
      )
      slug = _slugify(str(slug_raw).strip() or title or topic)
      raw_tags = obj.get("suggested_tags") or obj.get("tags") or meta_obj.get("suggested_tags") or []
      if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in re.split(r"[,;|]", raw_tags) if t.strip()]
      suggested_tags = [str(t).strip() for t in raw_tags if str(t).strip()]
      return {
        "metadata": {"title": title, "meta_description": _trim_meta(meta)},
        "keywords": keywords,
        "outline": outline,
        "content": {"article": article, "tone": tone},
        "faqs": faqs,
        "slug": slug,
        "suggested_tags": suggested_tags,
      }

  return _parse_markdown_article(raw, topic)


def _parse_markdown_article(raw: str, topic: str) -> dict[str, Any] | None:
  """Accept a plain markdown article when the model skips JSON."""
  text = (raw or "").strip()
  if not text:
    return None
  # Drop fenced code wrappers
  fence = re.search(r"```(?:markdown|md)?\s*([\s\S]*?)```", text, re.I)
  if fence:
    text = fence.group(1).strip()
  text = _strip_boilerplate_lines(_clean_body(text))
  if _count_words(text) < 120 or _is_boilerplate_article(text):
    return None

  title = topic.strip().title()[:70]
  m = re.match(r"^#\s+(.+)$", text, re.M)
  if m:
    title = m.group(1).strip()[:70]
  plain = re.sub(r"[#*_>`\[\]()]", "", text)
  meta = _trim_meta(plain[:200])
  article = seo_content_engine.strip_faq_section(text)
  outline = [{"level": "h2", "text": t} for t in seo_content_engine.extract_outline_from_body(article)]
  faqs = seo_content_engine.extract_faqs_from_body(article)
  return {
    "metadata": {"title": title, "meta_description": meta},
    "keywords": {"primary": topic, "secondary": []},
    "outline": outline,
    "content": {"article": article, "tone": ""},
    "faqs": faqs,
  }


def _outline_headings(template: dict[str, Any]) -> list[str]:
  outline = template.get("outline") or []
  heads: list[str] = []
  for item in outline:
    if isinstance(item, dict) and item.get("text"):
      level = str(item.get("level") or "h2").lower()
      if level in ("h2", "h3"):
        heads.append(str(item["text"]).strip())
    elif isinstance(item, str) and item.strip():
      heads.append(item.strip())
  return heads[:12]


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
  if ai.get("slug"):
    merged["slug"] = _slugify(str(ai["slug"]))
  if ai.get("suggested_tags"):
    merged["suggested_tags"] = [str(t).strip() for t in ai["suggested_tags"] if str(t).strip()]
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
  pipeline_meta: dict[str, Any] | None = None,
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
  slug = _slugify(str(structured.get("slug") or title or topic))
  suggested_tags = _suggest_tags(
    topic=topic,
    title=title,
    primary=primary,
    secondary=secondary,
    outline=outline_struct,
    category=category,
    ai_tags=structured.get("suggested_tags") if isinstance(structured.get("suggested_tags"), list) else None,
  )

  quality = pipeline_meta.get("quality") if pipeline_meta else None
  if not quality:
    quality = seo_content_engine.quality_report(title, meta_desc, article, keywords_flat)
  result = {
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
    "slug": slug,
    "suggested_tags": suggested_tags,
    "word_count": _count_words(article),
    "quality": quality,
    "discovery": discovery_meta,
    "ai": {"enabled": use_ai, "model_used": ai_used},
    "variation_seed": structured.get("variation_seed"),
    "domain": structured.get("domain"),
    "eeat": structured.get("eeat"),
    "keyword_density": structured.get("keyword_density"),
    "generator_version": GENERATOR_VERSION,
  }
  if pipeline_meta:
    result["architecture"] = pipeline_meta.get("architecture")
    result["pipeline_stages"] = pipeline_meta.get("stages")
    result["elapsed_ms"] = pipeline_meta.get("elapsed_ms")
    result["intent"] = pipeline_meta.get("intent")

    result["snippet"] = structured.get("snippet")
    result["schema"] = structured.get("schema")
  return result


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
  """Write a real SEO article via the model (hosted LLM / custom) — not template rewrite."""
  tone_guide = seo_content_engine.tone_hint(tone)
  headings = _outline_headings(template)
  outline_block = "\n".join(f"- {h}" for h in headings) if headings else (
    f"- What is {topic}?\n- Benefits and uses\n- Best practices\n"
    f"- Tips for beginners\n- Common mistakes\n- Conclusion"
  )
  evidence_block = (evidence_context or "").strip()[:3200] or (
    f"Use widely known, accurate facts about {topic}. Do not invent citations or URLs."
  )

  system_prompt = (
    f"You are Nexus, an expert SEO content writer ({tone} tone — {tone_guide}). "
    "Write ORIGINAL, useful, specific educational content a human would publish. "
    "Never mention Groq, Llama, OpenAI, Gemini, or any underlying model. "
    "Never use filler phrases like 'When exploring…', 'plays a key role in successful…', "
    "'focus on verified information', or 'Apply industry best practices and measure results'. "
    "Do not invent fake internal links or guide URLs. "
    f"Category: {category.replace('_', ' ')}. Target ~{target} words.{lang_line}{audience_line} "
    f"Preferred structure: {structure}. "
    "Return ONLY valid JSON (no markdown fences):\n"
    '{"metadata":{"title":"SEO title under 60 chars","meta_description":"155 chars max"},'
    '"keywords":{"primary":"...","secondary":["..."]},'
    '"outline":[{"level":"h2","text":"..."}],'
    '"content":{"article":"full markdown article starting with # Title — NO FAQ section","tone":"'
    + tone
    + '"},'
    '"faqs":[{"question":"...","answer":"..."}],'
    '"slug":"url-friendly-slug",'
    '"suggested_tags":["tag1","tag2","tag3"]}'
  )
  user_prompt = (
    f"Write a complete SEO article about: {topic}\n"
    f"Primary keyword: {primary}\n"
    f"Secondary keywords: {kw_line}\n\n"
    f"Suggested H2/H3 outline (you may improve it):\n{outline_block}\n\n"
    f"Evidence / facts to weave in naturally:\n{evidence_block}\n\n"
    "Requirements:\n"
    f"- Concrete advice specific to {topic} (definitions, how-to steps, tips, examples)\n"
    f"- Use the primary keyword naturally in title, intro, and 2–3 headings\n"
    "- Short paragraphs, bullet lists where helpful, one comparison table if useful\n"
    "- 4–6 FAQs with real answers\n"
    "- slug: lowercase hyphenated URL slug from the title (e.g. navio-coffee-guide)\n"
    "- suggested_tags: 6–12 short SEO tags (keywords + related topics, no filler)\n"
    "- No placeholder or meta text about 'generating SEO content'"
  )
  max_tokens = min(3500, max(1200, int(target * 2.2) + 400))
  raw = await asyncio.wait_for(
    provider.chat(
      [{"role": "user", "content": user_prompt}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=max_tokens,
      temperature=0.65,
    ),
    timeout=_AI_TIMEOUT_SEC,
  )
  parsed = _parse_structured_ai(raw, topic)
  if parsed:
    return parsed

  # Second pass: plain markdown if JSON failed
  md_system = (
    f"You are Nexus, an expert SEO writer ({tone}). "
    f"Write a complete markdown article (~{target} words) about {topic}. "
    "Start with # Title. Use ## headings. Be specific and useful. No filler. No FAQs section."
  )
  md_user = (
    f"Topic: {topic}\nKeywords: {primary}, {kw_line}\n"
    f"Outline:\n{outline_block}\n\nFacts:\n{evidence_block[:1600]}\n\n"
    "Write the full article now in markdown only."
  )
  raw2 = await asyncio.wait_for(
    provider.chat(
      [{"role": "user", "content": md_user}],
      system_prompt=md_system,
      use_rag=False,
      skip_intent=True,
      max_tokens=max_tokens,
      temperature=0.7,
    ),
    timeout=_AI_TIMEOUT_SEC,
  )
  return _parse_markdown_article(raw2, topic)


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
  """Structured SEO content — open-data RAG + hosted/custom model article writing."""
  topic = sanitize_topic(topic or "")
  if not topic:
    raise ValueError("topic is required")

  cat = seo_content_engine.normalize_category(category)
  tone_str = seo_content_engine.normalize_tone(tone, cat)
  lang_code = seo_content_engine.bcp47(language)
  target = max(150, min(1500, word_count or 500))

  kws = coerce_keywords(keywords)
  # If keywords accidentally contain the instruction-wrapped topic, drop them
  kws = [k for k in kws if k.lower() != topic.lower() and not _looks_like_instruction(k)]
  if topic.lower() not in {k.lower() for k in kws}:
    kws = [topic] + kws
  # Deduplicate while keeping order
  seen_kw: set[str] = set()
  deduped: list[str] = []
  for k in kws:
    if k.lower() not in seen_kw:
      seen_kw.add(k.lower())
      deduped.append(k)
  kws = deduped[:12]
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

  try:
    pipeline_out = await run_seo_content_pipeline(
      topic,
      kws,
      category=cat,
      tone=tone_str,
      audience=audience,
      target_words=target,
      variation_seed=seed,
      use_rag=use_rag,
    )
  except Exception:
    pipeline_out = None

  rag_meta: dict[str, Any] = {"enabled": use_rag, "confidence": 0.0, "sources_used": []}
  evidence_context = ""
  template: dict[str, Any]

  if pipeline_out:
    template = pipeline_out["structured"]
    rag_meta = pipeline_out.get("rag") or rag_meta
    evidence_context = pipeline_out.get("evidence_context") or ""
    seed = pipeline_out.get("variation_seed", seed)
  elif use_rag:
    try:
      from app.engine.seo_rag_pipeline import run_seo_rag_pipeline, synthesize_structured_content
      rag = await run_seo_rag_pipeline(topic, kws, category=cat, variation_seed=seed, top_k=8)
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
      if ai_result and not _is_boilerplate_article(ai_result["content"]["article"]):
        structured = _merge_ai_into_template(template, ai_result)
        # Prefer the AI article wholesale — do not keep template filler blocks
        structured["content"]["article"] = ai_result["content"]["article"]
        if ai_result["metadata"].get("title"):
          structured["metadata"]["title"] = ai_result["metadata"]["title"]
        if ai_result["metadata"].get("meta_description"):
          structured["metadata"]["meta_description"] = ai_result["metadata"]["meta_description"]
        if ai_result.get("faqs"):
          structured["faqs"] = ai_result["faqs"]
        if ai_result.get("outline"):
          structured["outline"] = ai_result["outline"]
        if ai_result.get("slug"):
          structured["slug"] = ai_result["slug"]
        if ai_result.get("suggested_tags"):
          structured["suggested_tags"] = ai_result["suggested_tags"]
        ai_used = True
    except Exception:
      structured = template

  # Last resort: scrub obvious filler lines from template output
  if not ai_used:
    art = structured.get("content", {}).get("article", "")
    cleaned = _strip_boilerplate_lines(art)
    if cleaned and _count_words(cleaned) >= 80:
      structured["content"]["article"] = cleaned
    structured["content"]["article"] = _strip_boilerplate_lines(
      structured["content"].get("article", "")
    )

  result = _pack_response(
    structured,
    topic=topic,
    category=cat,
    lang_code=lang_code,
    discovery_meta=discovery_meta,
    use_ai=use_ai,
    ai_used=ai_used,
    pipeline_meta=pipeline_out,
  )
  result["rag"] = rag_meta
  return result


# Backward-compatible alias
async def generate_seo_content(provider: ModelProvider | None, **kwargs) -> dict:
  return await generate(provider, **kwargs)

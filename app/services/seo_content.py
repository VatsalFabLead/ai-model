"""SEO Content Generator.

Creates SEO-optimized articles (title, meta description, slug, headings-rich
body) from a topic + keywords + tone, using the active free model backend.
No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import json
import re

from app.services.provider_base import ModelProvider

_VALID_TONES = {
  "professional", "casual", "friendly", "funny", "excited", "inspirational",
  "bold", "informative", "persuasive", "authoritative", "conversational", "neutral",
}


_GENERIC_HEADINGS = {
  "introduction", "intro", "overview", "summary", "contents", "table of contents",
  "getting started", "background", "conclusion", "about",
}


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "professional"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "professional"


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
  """Tidy small-model markdown: normalize headings, drop empty ones, collapse gaps."""
  cleaned: list[str] = []
  for ln in (body or "").split("\n"):
    m = re.match(r"^\s*(#{1,6})\s*(.*)$", ln)
    if m:
      level, rest = m.group(1), m.group(2)
      # Strip nested '###' markers and bold/italics that small models leak into headings.
      rest = re.sub(r"#{1,6}", "", rest).replace("**", "").strip().strip("*_`").strip()
      if not rest:
        continue
      cleaned.append(f"{level} {rest}")
    else:
      cleaned.append(ln)
  out = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
  return out.strip()


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


def _parse_seo(raw: str, topic: str) -> tuple[str, str, str]:
  """Return (title, meta_description, body_markdown)."""
  text = (raw or "").strip()

  # Some small models wrap output as JSON — handle that first.
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

  # Build the body by stripping the label lines and a leading "---" separator.
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


async def generate_seo_content(
  provider: ModelProvider,
  *,
  topic: str,
  keywords: list[str] | str | None = None,
  tone: str | None = None,
  word_count: int | None = None,
  audience: str | None = None,
) -> dict:
  tone_str = _normalize_tone(tone)
  kws = coerce_keywords(keywords)
  primary = kws[0] if kws else topic.strip()
  target = max(150, min(1500, word_count or 500))

  kw_line = ", ".join(kws) if kws else "(none given — pick relevant keywords yourself)"
  audience_line = f" Target audience: {audience.strip()}." if audience else ""

  system_prompt = (
    f"You are an expert SEO content writer and copywriter. Write a {tone_str}, "
    f"original, SEO-optimized article of about {target} words that ranks well in "
    f"search engines and genuinely engages readers. Rules: include the primary "
    f"keyword in the title and the first paragraph; use clear '##' and '###' markdown "
    f"headings; integrate the keywords naturally without stuffing; add a short intro "
    f"and a concluding takeaway.{audience_line} "
    f"Respond EXACTLY in this format and nothing else:\n"
    f"TITLE: <catchy SEO title>\n"
    f"META: <compelling meta description under 160 characters>\n"
    f"---\n"
    f"<the full article body in markdown>"
  )
  user_prompt = (
    f"Topic: {topic.strip()}\n"
    f"Primary keyword: {primary}\n"
    f"Keywords to include: {kw_line}"
  )

  max_tokens = min(1100, int(target * 1.6) + 80)
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    max_tokens=max_tokens,
    temperature=0.7,
  )

  title, meta, body = _parse_seo(raw, topic)
  body = _clean_body(body)
  return {
    "title": title,
    "meta_description": meta,
    "slug": _slugify(title),
    "keywords": kws,
    "content": body,
    "word_count": _count_words(body),
  }

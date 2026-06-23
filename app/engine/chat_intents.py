"""Detect specialized chat intents and format task-specific replies."""

from __future__ import annotations

import re


def is_title_meta_query(text: str) -> bool:
  low = (text or "").lower()
  if "meta description" in low or "meta desc" in low:
    return True
  if "seo title" in low:
    return True
  if "title" in low and "meta" in low:
    return True
  if re.search(r"\btitle\s*&\s*meta\b", low):
    return True
  return False


def extract_title_meta_topic(text: str) -> str:
  topic = (text or "").strip()
  topic = re.sub(
    r"\b(for\s+)?seo\s+(title|titles)(\s*&\s*meta(\s*description)?)?\b.*$",
    "",
    topic,
    flags=re.IGNORECASE,
  ).strip(" -:,")
  topic = re.sub(
    r"\b(title\s*&\s*meta(\s*description)?|meta\s*description)\b.*$",
    "",
    topic,
    flags=re.IGNORECASE,
  ).strip(" -:,")
  return topic or text.strip()


def format_title_meta_reply(result: dict) -> str:
  topic = result.get("topic") or "Topic"
  lines = [f"## SEO Title & Meta — {topic}\n"]
  for i, item in enumerate(result.get("variations") or [], 1):
    angle = item.get("angle") or f"option {i}"
    lines.append(f"### Option {i} ({angle})")
    lines.append(f"**Title** ({item.get('title_length', 0)} chars): {item.get('title', '')}")
    lines.append(
      f"**Meta description** ({item.get('meta_length', 0)} chars): "
      f"{item.get('meta_description', '')}"
    )
    score = item.get("quality_score")
    if score is not None:
      ready = "yes" if item.get("seo_ready") else "no"
      lines.append(f"_Quality: {score}/100 · SEO-ready: {ready}_")
    lines.append("")
  quality = result.get("quality") or {}
  avg = quality.get("average_score")
  if avg is not None:
    lines.append(f"**Average quality:** {avg}/100")
  return "\n".join(lines).strip()

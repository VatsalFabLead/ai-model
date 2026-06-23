"""AI helpers for the Post Scheduler tool.

Generates platform-aware post content and hashtag suggestions using whatever
ModelProvider is active (the free open-source LLM backend is recommended).
No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import json
import re

from app.services.provider_base import ModelProvider

# Per-platform writing rules. char_limit mirrors real platform limits.
PLATFORMS: dict[str, dict] = {
  "instagram": {
    "label": "Instagram",
    "char_limit": 2200,
    "hashtags": 12,
    "style": "engaging and visual; use short lines, a few tasteful emojis, and end with a call to action",
  },
  "twitter": {
    "label": "X (Twitter)",
    "char_limit": 280,
    "hashtags": 3,
    "style": "concise and punchy; it MUST fit within 280 characters",
  },
  "x": {
    "label": "X (Twitter)",
    "char_limit": 280,
    "hashtags": 3,
    "style": "concise and punchy; it MUST fit within 280 characters",
  },
  "facebook": {
    "label": "Facebook",
    "char_limit": 2000,
    "hashtags": 3,
    "style": "warm and conversational; encourage comments and shares",
  },
  "linkedin": {
    "label": "LinkedIn",
    "char_limit": 3000,
    "hashtags": 5,
    "style": "professional and insightful; use short paragraphs and a value-driven hook",
  },
  "tiktok": {
    "label": "TikTok",
    "char_limit": 2200,
    "hashtags": 6,
    "style": "fun, trendy, and energetic; strong hook in the first line",
  },
  "youtube": {
    "label": "YouTube",
    "char_limit": 5000,
    "hashtags": 5,
    "style": "descriptive; summarize the video and invite likes/subscribes",
  },
  "threads": {
    "label": "Threads",
    "char_limit": 500,
    "hashtags": 3,
    "style": "casual and conversational",
  },
  "pinterest": {
    "label": "Pinterest",
    "char_limit": 500,
    "hashtags": 5,
    "style": "descriptive and keyword-rich to aid discovery",
  },
}

DEFAULT_PLATFORM = {
  "label": "Social Media",
  "char_limit": 2200,
  "hashtags": 8,
  "style": "clear, engaging, and audience-friendly",
}

_VALID_TONES = {
  "professional", "casual", "friendly", "funny", "excited",
  "inspirational", "bold", "informative", "persuasive", "neutral",
}

_STOPWORDS = {
  "the", "and", "for", "with", "your", "you", "our", "are", "this", "that",
  "from", "have", "has", "will", "into", "out", "about", "what", "when",
  "how", "why", "who", "a", "an", "of", "to", "in", "on", "is", "it", "we",
  "be", "or", "as", "at", "by", "my", "me", "i", "new", "get", "all",
}


def platform_config(platform: str | None) -> dict:
  if not platform:
    return DEFAULT_PLATFORM
  return PLATFORMS.get(platform.strip().lower(), DEFAULT_PLATFORM)


def supported_platforms() -> list[dict]:
  seen: set[str] = set()
  out: list[dict] = []
  for key, cfg in PLATFORMS.items():
    if cfg["label"] in seen:
      continue
    seen.add(cfg["label"])
    out.append({
      "id": key,
      "label": cfg["label"],
      "char_limit": cfg["char_limit"],
      "recommended_hashtags": cfg["hashtags"],
    })
  return out


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "engaging"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "engaging"


def _unwrap_json_like(text: str) -> str:
  """Small models sometimes wrap the post in JSON. Extract the real text."""
  t = text.strip()
  if not t.startswith("{"):
    return text
  try:
    obj = json.loads(t)
    if isinstance(obj, dict):
      for key in ("content", "post", "caption", "text", "body", "message"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
          return val.strip()
  except Exception:
    pass
  # Truncated/loose JSON: grab the content field through to the end.
  m = re.search(r'"(?:content|post|caption|text|body)"\s*:\s*"(.*)', t, re.DOTALL)
  if m:
    val = m.group(1).rstrip()
    if val.endswith('"'):
      val = val[:-1]
    val = val.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
    return val.strip()
  return text


def _clean_generated(text: str) -> str:
  """Strip common small-model artifacts: JSON wrap, preambles, labels, quotes."""
  text = _unwrap_json_like((text or "").strip()).strip()
  # Drop a leading "Sure, here's ...:" / "Here is ...:" preamble line.
  text = re.sub(
    r"^(sure[,!.]?\s+)?(here(?:'s| is| are)|certainly|of course)[^\n:]*:\s*",
    "",
    text,
    flags=re.IGNORECASE,
  )
  # Drop a leading label like "Post:" / "Caption:" / "Content:".
  text = re.sub(r"^(post|caption|content|tweet)\s*:\s*", "", text, flags=re.IGNORECASE)
  text = text.strip()
  # Strip a single pair of wrapping quotes.
  if len(text) >= 2 and text[0] in "\"'\u201c\u2018" and text[-1] in "\"'\u201d\u2019":
    text = text[1:-1].strip()
  return text


def _trim_to_limit(text: str, limit: int) -> str:
  if len(text) <= limit:
    return text
  cut = text[:limit]
  # Avoid chopping mid-word when possible.
  space = cut.rfind(" ")
  if space > limit * 0.6:
    cut = cut[:space]
  return cut.rstrip() + "\u2026"


def _dedupe_hashtags(tags: list[str], count: int) -> list[str]:
  seen: set[str] = set()
  out: list[str] = []
  for h in tags:
    h = h.strip()
    # Drop too-short and run-on/garbage tags from small models.
    if len(h) < 2 or len(h) > 30:
      continue
    key = h.lower()
    if key in seen:
      continue
    seen.add(key)
    out.append(h)
    if len(out) >= count:
      break
  return out


def _camel_tag(word: str) -> str:
  return "#" + word[0].upper() + word[1:] if word else ""


def _hashtags_from_topic(topic: str, count: int) -> list[str]:
  words = re.findall(r"[A-Za-z0-9]{3,}", topic)
  tags = [_camel_tag(w) for w in words if w.lower() not in _STOPWORDS]
  return _dedupe_hashtags(tags, count)


def _parse_hashtags(text: str, topic: str, count: int) -> list[str]:
  found = re.findall(r"#\w+", text or "")
  if not found:
    # Model returned plain words — turn them into hashtags.
    words = re.findall(r"[A-Za-z0-9]{2,}", text or "")
    found = [_camel_tag(w) for w in words if w.lower() not in _STOPWORDS]
  tags = _dedupe_hashtags(found, count)
  if len(tags) < count:
    # Top up from the topic so the frontend always gets enough.
    tags = _dedupe_hashtags(tags + _hashtags_from_topic(topic, count), count)
  return tags


async def suggest_content(
  provider: ModelProvider,
  *,
  platform: str,
  topic: str,
  tone: str | None = None,
  keywords: list[str] | None = None,
  include_emojis: bool = True,
  include_hashtags: bool = False,
) -> dict:
  cfg = platform_config(platform)
  tone_str = _normalize_tone(tone)
  emoji_rule = (
    "Include a few relevant emojis." if include_emojis else "Do not use emojis."
  )
  hashtag_rule = (
    f"End with up to {cfg['hashtags']} relevant hashtags."
    if include_hashtags
    else "Do NOT include any hashtags."
  )

  system_prompt = (
    f"You are an expert social media manager and copywriter for {cfg['label']}. "
    f"Write a single {tone_str} post. Style: {cfg['style']}. {emoji_rule} {hashtag_rule} "
    f"Keep it well under {cfg['char_limit']} characters. "
    f"Output ONLY the post text as plain text — never JSON, no preamble, "
    f"no explanations, no surrounding quotes."
  )

  user_lines = [f"Topic: {topic.strip()}"]
  if keywords:
    kw = ", ".join(k.strip() for k in keywords if k.strip())
    if kw:
      user_lines.append(f"Keywords to include: {kw}")
  user_prompt = "\n".join(user_lines)

  max_tokens = max(80, min(512, cfg["char_limit"] // 3))
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    max_tokens=max_tokens,
    temperature=0.8,
  )
  content = _trim_to_limit(_clean_generated(raw), cfg["char_limit"])
  return {
    "platform": cfg["label"],
    "content": content,
    "char_count": len(content),
    "char_limit": cfg["char_limit"],
  }


async def suggest_hashtags(
  provider: ModelProvider,
  *,
  platform: str,
  topic: str,
  count: int | None = None,
) -> dict:
  cfg = platform_config(platform)
  n = count or cfg["hashtags"]
  n = max(1, min(30, n))

  system_prompt = (
    f"You are a {cfg['label']} hashtag expert. Generate exactly {n} relevant, "
    f"popular, discoverable hashtags for the given topic. "
    f"Return ONLY the hashtags separated by single spaces, each starting with '#'. "
    f"No numbering, no explanations, no other text."
  )
  raw = await provider.chat(
    [{"role": "user", "content": f"Topic/content: {topic.strip()}"}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    max_tokens=120,
    temperature=0.6,
  )
  tags = _parse_hashtags(raw, topic, n)
  return {
    "platform": cfg["label"],
    "hashtags": tags,
    "text": " ".join(tags),
  }


async def generate_post(
  provider: ModelProvider,
  *,
  platform: str,
  topic: str,
  tone: str | None = None,
  keywords: list[str] | None = None,
  include_emojis: bool = True,
  hashtag_count: int | None = None,
) -> dict:
  content = await suggest_content(
    provider,
    platform=platform,
    topic=topic,
    tone=tone,
    keywords=keywords,
    include_emojis=include_emojis,
    include_hashtags=False,
  )
  tags = await suggest_hashtags(
    provider, platform=platform, topic=topic, count=hashtag_count
  )
  return {
    "platform": content["platform"],
    "content": content["content"],
    "char_count": content["char_count"],
    "char_limit": content["char_limit"],
    "hashtags": tags["hashtags"],
    "hashtags_text": tags["text"],
  }

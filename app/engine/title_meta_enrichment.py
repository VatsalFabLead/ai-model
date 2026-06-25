"""Title & Meta enrichment — natural copy, no retrieval leakage, differentiated scoring."""

from __future__ import annotations

import re
from typing import Any

TITLE_MAX = 60
META_MIN = 140
META_MAX = 160

_POLLUTION_MARKERS = (
  "## ",
  "food delivery is a courier",
  "wikipedia",
  "is a courier service",
  "according to",
  "was founded in",
  "is a city in",
  "is a country",
)

_AWKWARD_TITLE_PATTERNS = (
  r"top\s+\w+\s+guide\s+to",
  r"from zero to",
  r"you can use$",
  r"why\s+.+\s+matters in",
  r"^\w+\s+guide\s+to\s+\1",  # won't use backref in simple check
)

_SERP_PATTERNS = (
  "ultimate_guide", "step_by_step", "complete_guide", "best_practices",
  "cost_breakdown", "examples", "checklist", "how_to", "comparison",
  "numbered_list", "year_stamped", "question_title", "colon_subtitle",
)

_FEATURE_BITS: dict[str, list[str]] = {
  "app": ["key features", "development costs", "technology stack", "best practices"],
  "software": ["features", "pricing", "implementation", "best practices"],
  "marketing": ["strategies", "conversion tips", "examples", "best practices"],
  "default": ["essential tips", "expert insights", "practical steps", "best practices"],
}


def _clip(text: str, n: int) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= n else t[: n - 3].rstrip() + "..."


def normalize_topic_phrase(topic: str) -> str:
  t = re.sub(r"\s+", " ", (topic or "").strip())
  t = re.sub(r"^(how to|what is|guide to)\s+", "", t, flags=re.I)
  return t.strip() or "your topic"


def topic_display(topic: str) -> str:
  t = normalize_topic_phrase(topic)
  if not t:
    return "Your Topic"
  return " ".join(w.capitalize() for w in t.split())


def detect_topic_profile(topic: str) -> str:
  low = topic.lower()
  if "app" in low or "software" in low or "saas" in low:
    return "app"
  if any(w in low for w in ("marketing", "seo", "email", "ads")):
    return "marketing"
  if any(w in low for w in ("api", "framework", "developer", "programming")):
    return "software"
  return "default"


def extract_keywords_enhanced(topic: str) -> dict[str, Any]:
  phrase = normalize_topic_phrase(topic)
  low = phrase.lower()
  words = [w for w in re.findall(r"\w+", phrase) if len(w) > 2]
  primary = phrase
  secondary = words[:4] if len(words) > 1 else []

  profile = detect_topic_profile(phrase)
  long_tail: list[str] = []
  if profile == "app":
    base = low if low.endswith("app") else f"{low} app"
    long_tail = [
      f"how to build a {base}",
      f"{base} development",
      f"best {base} features",
      f"{base} cost",
      f"{base} development guide",
      f"{base} technology stack",
    ]
  elif low.startswith("how to"):
    long_tail = [
      phrase,
      f"{low} guide",
      f"{low} step by step",
      f"best practices for {low.replace('how to ', '')}",
    ]
  else:
    long_tail = [
      f"how to {low}" if not low.startswith("how") else phrase,
      f"best {low}",
      f"{low} guide",
      f"{low} tips",
      f"{low} best practices",
      f"complete {low} guide",
    ]

  lsi = list(dict.fromkeys(words + [w.lower() for w in words if len(w) > 4]))[:10]
  return {
    "primary": primary,
    "secondary": secondary,
    "long_tail": list(dict.fromkeys(long_tail))[:8],
    "lsi": lsi,
    "profile": profile,
  }


def detect_intent_extended(topic: str, keywords: dict[str, Any], category: str) -> dict[str, Any]:
  low = topic.lower()
  scores = {
    "informational": sum(1 for w in ("how", "what", "why", "guide", "learn", "tips", "explained") if w in low),
    "commercial": sum(1 for w in ("best", "top", "review", "vs", "compare", "cost", "pricing") if w in low),
    "transactional": sum(1 for w in ("buy", "price", "shop", "download", "hire") if w in low),
    "navigational": sum(1 for w in ("official", "login", "website") if w in low),
  }
  primary = max(scores, key=scores.get)
  if scores[primary] == 0:
    primary = "informational"

  content_type = "guide"
  if category in ("how_to",) or low.startswith("how to") or "how to" in low:
    content_type = "how_to_guide"
  elif category in ("product_page", "ecommerce"):
    content_type = "product"
  elif category in ("landing_page", "saas"):
    content_type = "landing"
  elif category in ("local_business",):
    content_type = "local"
  elif "app" in low:
    content_type = "development_guide"

  audience = "general"
  if any(w in low for w in ("developer", "development", "api", "code", "programming", "app")):
    audience = "developers"
  elif any(w in low for w in ("business", "startup", "enterprise", "saas")):
    audience = "business"
  elif any(w in low for w in ("beginner", "starter", "intro")):
    audience = "beginners"

  serp_intent = "educational"
  if primary == "commercial":
    serp_intent = "comparison"
  elif primary == "transactional":
    serp_intent = "conversion"
  elif content_type == "how_to_guide":
    serp_intent = "instructional"

  ctr_pattern = "guide"
  if content_type == "how_to_guide" or low.startswith("how"):
    ctr_pattern = "how_to"
  elif "cost" in low or "pricing" in low:
    ctr_pattern = "cost_breakdown"
  elif primary == "commercial":
    ctr_pattern = "best_list"
  elif "checklist" in low:
    ctr_pattern = "checklist"

  return {
    "primary": primary,
    "scores": scores,
    "content_type": content_type,
    "audience": audience,
    "serp_intent": serp_intent,
    "ctr_pattern": ctr_pattern,
  }


def analyze_serp_patterns_extended(docs: list[Any], topic: str) -> dict[str, Any]:
  patterns: dict[str, int] = {p: 0 for p in _SERP_PATTERNS}
  samples: list[str] = []
  for d in docs[:15]:
    title = (getattr(d, "title", None) or "").strip()
    if not title:
      continue
    samples.append(title[:90])
    tl = title.lower()
    if "ultimate" in tl and "guide" in tl:
      patterns["ultimate_guide"] += 1
    if "step" in tl and ("step" in tl or "by step" in tl):
      patterns["step_by_step"] += 1
    if "complete guide" in tl or "complete" in tl and "guide" in tl:
      patterns["complete_guide"] += 1
    if "best practice" in tl:
      patterns["best_practices"] += 1
    if "cost" in tl or "pricing" in tl:
      patterns["cost_breakdown"] += 1
    if "example" in tl:
      patterns["examples"] += 1
    if "checklist" in tl:
      patterns["checklist"] += 1
    if title.startswith("How to") or title.startswith("How To"):
      patterns["how_to"] += 1
    if " vs " in tl or "versus" in tl:
      patterns["comparison"] += 1
    if re.search(r"\b\d+\b", title):
      patterns["numbered_list"] += 1
    if re.search(r"\b(19|20)\d{2}\b", title):
      patterns["year_stamped"] += 1
    if "?" in title:
      patterns["question_title"] += 1
    if ":" in title:
      patterns["colon_subtitle"] += 1

  ranked = sorted(patterns, key=patterns.get, reverse=True)
  recommended = [p for p in ranked if patterns[p] > 0][:6]
  if not recommended:
    profile = detect_topic_profile(topic)
    recommended = (
      ["complete_guide", "how_to", "best_practices", "cost_breakdown"]
      if profile == "app"
      else ["complete_guide", "best_practices", "how_to"]
    )
  return {"patterns": patterns, "samples": samples[:6], "recommended": recommended}


def sanitize_facts_from_docs(docs: list[Any], topic: str) -> list[str]:
  """Extract clean fact phrases — never headings or raw wiki intros."""
  anchors = set(re.findall(r"\w+", topic.lower()))
  facts: list[str] = []
  seen: set[str] = set()
  for d in docs[:8]:
    text = re.sub(r"^#+\s*.+$", "", getattr(d, "text", "") or "", flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    for sent in re.split(r"(?<=[.!?])\s+", text):
      sent = sent.strip()
      if len(sent) < 45 or len(sent) > 220:
        continue
      if sent.startswith("#") or "##" in sent:
        continue
      low = sent.lower()
      if any(m in low for m in _POLLUTION_MARKERS):
        continue
      if anchors and not any(a in low for a in anchors if len(a) > 3):
        continue
      key = low[:80]
      if key in seen:
        continue
      seen.add(key)
      facts.append(sent)
  return facts[:6]


def is_polluted_metadata(text: str) -> bool:
  low = (text or "").lower()
  if "##" in text or text.strip().startswith("#"):
    return True
  if any(m in low for m in _POLLUTION_MARKERS):
    return True
  if re.search(r"\b(is a|are a|was a)\s+\w+\s+(service|city|country|company)\b", low):
    return True
  return False


def is_awkward_title(title: str) -> bool:
  low = title.lower()
  if re.search(r"top\s+\w+\s+guide\s+to", low):
    return True
  if "from zero to" in low:
    return True
  if low.endswith("you can use"):
    return True
  if re.search(r"why\s+.+\s+matters in\s+\d{4}", low):
    return True
  if re.search(r"\bapp\s+app\b", low):
    return True
  if re.search(r"\bhow to\s+how to\b", low):
    return True
  return False


def build_title(ctx: dict[str, Any], idx: int) -> tuple[str, str]:
  """Natural, keyword-forward titles."""
  topic = ctx["topic_display"]
  phrase = ctx["phrase"]
  year = ctx.get("year", "2026")
  profile = ctx.get("profile", "default")
  intent = ctx.get("intent", {})
  serp = ctx.get("serp", {}).get("recommended", ["complete_guide"])
  salt = ctx["seed"] + idx * 41

  def pick(pool: list[tuple[str, str]], i: int) -> tuple[str, str]:
    return pool[i % len(pool)]

  app_titles = [
    (f"{topic} Development Guide: Features, Cost & Best Practices ({year})", "complete_guide"),
    (f"How to Build a {topic}: Complete Guide ({year})", "how_to"),
    (f"{topic}: Technology Stack, Features & Costs ({year})", "cost_breakdown"),
    (f"Build a {topic}: Step-by-Step Guide ({year})", "step_by_step"),
    (f"{topic} - Ultimate Development Guide ({year})", "ultimate_guide"),
    (f"{topic} Cost & Features: Expert Guide ({year})", "cost_breakdown"),
  ]
  general_titles = [
    (f"{topic}: Complete Guide & Best Practices ({year})", "complete_guide"),
    (f"How to Master {topic}: Step-by-Step Guide ({year})", "how_to"),
    (f"{topic} — Expert Tips, Examples & Checklist ({year})", "checklist"),
    (f"The Ultimate {topic} Guide ({year})", "ultimate_guide"),
    (f"{topic}: Everything You Need to Know ({year})", "complete_guide"),
    (f"Best {topic} Strategies for {year}", "best_practices"),
  ]

  pool = app_titles if profile == "app" else general_titles

  if intent.get("ctr_pattern") == "how_to":
    pool = sorted(pool, key=lambda x: 0 if "how" in x[0].lower() else 1)
  if "cost_breakdown" in serp:
    pool = sorted(pool, key=lambda x: 0 if "cost" in x[0].lower() else 1)

  title, angle = pick(pool, salt)
  if idx % 5 == 2 and "checklist" in serp and profile != "app":
    title = f"{topic}: Practical Examples & Checklist ({year})"
    angle = "checklist"

  return title, angle


def build_meta_description(ctx: dict[str, Any], title: str, idx: int) -> str:
  """CTR-focused meta — synthesized, never raw retrieval."""
  phrase = ctx["phrase"]
  low = phrase.lower()
  profile = ctx.get("profile", "default")
  bits = _FEATURE_BITS.get(profile, _FEATURE_BITS["default"])
  salt = ctx["seed"] + idx * 17

  if profile == "app":
    templates = [
      (
        f"Learn how to build a {low} with {bits[0]}, {bits[1]}, {bits[2]}, and {bits[3]}. "
        "Start planning your project today."
      ),
      (
        f"Discover how to plan, design, and launch a {low} with expert tips on features, costs, "
        "and technology choices. Read the complete guide."
      ),
      (
        f"Explore {low} development with practical guidance on features, pricing, tech stack, "
        "and launch strategy. Get started with confidence."
      ),
    ]
  else:
    templates = [
      (
        f"Learn {low} with {bits[0]}, {bits[1]}, and {bits[2]} explained clearly. "
        "Read expert tips and start improving results today."
      ),
      (
        f"Discover proven strategies for {low} with actionable steps, examples, and best practices. "
        "Explore the full guide now."
      ),
      (
        f"Get a complete overview of {low} covering practical tips, common mistakes, and expert advice. "
        "Start learning today."
      ),
    ]

  meta = templates[salt % len(templates)]
  tone = ctx.get("tone", "professional")
  if tone == "casual":
    meta = f"Want the real scoop on {low}? Practical tips, no fluff — plus {bits[0]} and {bits[1]}. Dive in now."
  elif tone == "formal":
    meta = (
      f"A comprehensive overview of {low}, including {bits[0]}, {bits[1]}, and evidence-based recommendations. "
      "Review the full analysis."
    )

  return _clip(meta, META_MAX)


def validate_metadata_pair(title: str, meta: str, topic: str) -> dict[str, Any]:
  issues: list[str] = []
  tl, ml = len(title), len(meta)
  topic_l = normalize_topic_phrase(topic).lower()
  title_l = title.lower()

  if is_polluted_metadata(meta) or is_polluted_metadata(title):
    issues.append("source_leakage")
  if "##" in meta or meta.strip().startswith("#"):
    issues.append("markdown_in_meta")
  if is_awkward_title(title):
    issues.append("awkward_title")
  if tl > TITLE_MAX:
    issues.append("title_too_long")
  elif tl < 40:
    issues.append("title_too_short")
  if ml > META_MAX:
    issues.append("meta_too_long")
  elif ml < META_MIN:
    issues.append("meta_too_short")
  if topic_l and topic_l.split()[0] not in title_l and not any(
    w in title_l for w in topic_l.split()[:2] if len(w) > 3
  ):
    issues.append("keyword_missing_in_title")
  if topic_l.split()[0] not in meta.lower():
    issues.append("keyword_missing_in_meta")
  if meta.endswith("...") and ml < META_MIN + 10:
    issues.append("truncated_meta")
  if not re.search(r"[.!?]$", meta.strip()):
    issues.append("meta_no_terminal_punctuation")
  if title_l == meta.lower()[: min(len(title_l), len(meta))]:
    issues.append("duplicate_wording")

  return {"issues": issues, "valid": not issues}


def score_metadata_pair(
  title: str,
  meta: str,
  topic: str,
  intent: dict[str, Any],
  idx: int,
) -> dict[str, Any]:
  """Differentiated SEO, CTR, and overall scores."""
  validation = validate_metadata_pair(title, meta, topic)
  issues = validation["issues"]

  seo = 88
  ctr = 85
  topic_l = normalize_topic_phrase(topic).lower()
  tl, ml = len(title), len(meta)

  if 48 <= tl <= TITLE_MAX:
    seo += 4
  elif tl < 40:
    seo -= 8
  if META_MIN <= ml <= META_MAX:
    seo += 4
  elif ml < META_MIN:
    seo -= 10

  if topic_l and title.lower().startswith(topic_l.split()[0]):
    seo += 5
    ctr += 4
  elif topic_l in title.lower():
    seo += 3

  if ":" in title:
    ctr += 3
  if re.search(r"\b(20\d{2})\b", title):
    ctr += 2
  if any(w in title.lower() for w in ("guide", "complete", "how to", "best")):
    ctr += 2
  if any(w in meta.lower() for w in ("learn", "discover", "start", "today", "guide")):
    ctr += 3
  if "how to build" in meta.lower() or "get started" in meta.lower():
    ctr += 4

  penalty = {
    "source_leakage": 25,
    "markdown_in_meta": 20,
    "awkward_title": 18,
    "title_too_long": 12,
    "title_too_short": 8,
    "meta_too_long": 10,
    "meta_too_short": 12,
    "keyword_missing_in_title": 10,
    "keyword_missing_in_meta": 8,
    "truncated_meta": 6,
    "duplicate_wording": 8,
  }
  for issue in issues:
    seo -= penalty.get(issue, 4)
    ctr -= max(2, penalty.get(issue, 4) - 2)

  # Variation spread so not every option scores 100
  seo -= (idx % 5) * 2
  ctr -= (idx % 4) * 2

  seo = max(55, min(99, seo))
  ctr = max(55, min(99, ctr))
  overall = round(seo * 0.55 + ctr * 0.45)

  return {
    "seo_score": seo,
    "ctr_score": ctr,
    "overall_score": overall,
    "quality_score": overall,
    "seo_ready": overall >= 75 and "source_leakage" not in issues,
    "issues": issues,
  }


def trim_title(title: str) -> str:
  title = re.sub(r"\s+", " ", (title or "").strip())
  if len(title) <= TITLE_MAX:
    return title
  cut = title[:TITLE_MAX].rsplit(" ", 1)[0]
  return cut.rstrip(" -:|,")


def trim_meta(meta: str) -> str:
  meta = re.sub(r"\s+", " ", (meta or "").strip())
  if len(meta) > META_MAX:
    meta = meta[: META_MAX - 3].rsplit(" ", 1)[0] + "..."
  if len(meta) < META_MIN:
    meta = _clip(meta + " Explore expert tips and start today.", META_MAX)
  return meta

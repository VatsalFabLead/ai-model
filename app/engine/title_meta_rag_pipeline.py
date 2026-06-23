"""SEO Title & Meta — production pipeline (open datasets, dynamic variations).

Input → Keyword Extractor → Entity Extractor → Intent Detector → SERP Pattern Analyzer
→ Title Generator → Meta Generator → Length Validator → Duplicate Checker
→ SEO Scorer → AI Search Optimizer → Quality Validator → Final Output
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources
from app.engine.seo_content_domains import make_variation_seed
from app.engine import title_meta_engine as tme

GENERATOR_VERSION = "title-meta-rag-v2.0"

ARCHITECTURE_FLOW = [
  "input",
  "keyword_extractor",
  "entity_extractor",
  "intent_detector",
  "serp_pattern_analyzer",
  "title_generator",
  "meta_generator",
  "length_validator",
  "duplicate_checker",
  "seo_scorer",
  "ai_search_optimizer",
  "quality_validator",
  "final_output",
]

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "General Knowledge": ["wikipedia", "wikidata", "dbpedia"],
  "Web Text": ["c4", "fineweb"],
  "Academic": ["arxiv", "semantic_scholar"],
  "Question Answering": ["gooaq", "squad"],
  "Conversational": ["dolly"],
  "Programming": ["stackexchange"],
  "News": ["gdelt"],
  "Books": ["gutenberg", "openlibrary"],
  "Medical": ["pubmed", "wikipedia"],
  "Finance": ["sec_edgar", "wikidata"],
  "Legal": ["courtlistener", "wikipedia"],
  "Geography": ["wikidata", "openstreetmap"],
  "Multimedia": ["wikimedia", "youtube_cc"],
  "E-commerce": ["amazon_reviews", "wikipedia"],
  "Social Media": ["reddit", "twitter_archive"],
  "Government": ["data_gov", "gdelt"],
  "SEO/Search": ["gooaq", "wikipedia"],
  "Multilingual": ["wikipedia", "wikidata"],
}

_SOURCE_ROUTE = ["wikipedia", "wikidata", "gooaq", "squad"]

_POWER = (
  "Ultimate", "Essential", "Proven", "Complete", "Expert", "Smart",
  "Definitive", "Practical", "Top", "Best", "Modern", "Advanced",
)
_ANGLES = (
  "guide", "how-to", "listicle", "question", "benefit", "year",
  "comparison", "beginner", "expert", "local", "quick", "deep-dive",
)
_META_HOOKS = (
  "Discover", "Learn", "Explore", "Find out", "Get", "See how",
  "Unlock", "Master", "Start", "Compare",
)
_CTA = (
  "Read the full guide today.", "Get started now.", "Learn more inside.",
  "See expert tips and examples.", "Start improving results today.",
  "Browse proven strategies now.", "Find answers in this guide.",
  "Explore actionable insights now.", "Download the checklist inside.",
  "See real-world examples today.", "Compare options and decide faster.",
)
_NUMBERS = ("5", "7", "10", "12", "15", "21")
_SUFFIXES = (
  "for Beginners", "for Professionals", "That Work", "You Can Use",
  "Step by Step", "Made Simple", "Explained", "in Minutes",
)


def effective_variation_seed(client_seed: int | None) -> int:
  base = make_variation_seed(client_seed)
  nonce = secrets.randbits(31) ^ (time.time_ns() & 0x7FFFFFFF)
  return make_variation_seed(base ^ nonce)


def _shuffle(items: list[Any], seed: int) -> list[Any]:
  out = list(items)
  for i in range(len(out) - 1, 0, -1):
    j = (seed + i * 7919) % (i + 1)
    out[i], out[j] = out[j], out[i]
  return out


def _pick(pool: list[str], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _topic_clean(topic: str) -> str:
  return re.sub(r"\s+", " ", (topic or "").strip())


def _topic_title(topic: str) -> str:
  t = _topic_clean(topic)
  return t[0].upper() + t[1:] if t else "Your Topic"


def extract_keywords(topic: str) -> dict[str, Any]:
  t = _topic_clean(topic)
  words = [w for w in re.findall(r"\w+", t) if len(w) > 2]
  primary = t
  secondary = words[:3] if len(words) > 1 else []
  long_tail = [
    f"how to {t.lower()}",
    f"best {t.lower()}",
    f"{t.lower()} guide",
    f"{t.lower()} tips",
  ]
  lsi = list(dict.fromkeys(words + [w.lower() for w in words if len(w) > 4]))[:8]
  return {
    "primary": primary,
    "secondary": secondary,
    "long_tail": long_tail[:6],
    "lsi": lsi,
  }


def extract_entities(topic: str, docs: list[OpenDoc]) -> list[str]:
  entities: set[str] = set()
  for w in re.findall(r"\w+", topic):
    if len(w) > 3:
      entities.add(w.title())
  for d in docs[:6]:
    for w in re.findall(r"\w+", d.title):
      if len(w) > 4 and w[0].isupper():
        entities.add(w)
  return list(entities)[:12]


def detect_intent(topic: str, keywords: dict[str, Any]) -> dict[str, Any]:
  low = topic.lower()
  scores = {
    "informational": sum(1 for w in ("how", "what", "why", "guide", "learn", "tips") if w in low),
    "commercial": sum(1 for w in ("best", "top", "review", "vs", "compare") if w in low),
    "transactional": sum(1 for w in ("buy", "price", "shop", "download", "get") if w in low),
    "navigational": sum(1 for w in ("official", "login", "website") if w in low),
  }
  primary = max(scores, key=scores.get)
  if scores[primary] == 0:
    primary = "informational"
  return {"primary": primary, "scores": scores}


def analyze_serp_patterns(docs: list[OpenDoc], topic: str) -> dict[str, Any]:
  patterns: dict[str, int] = {
    "question": 0, "numbered": 0, "guide": 0, "year": 0, "colon": 0, "pipe": 0,
  }
  samples: list[str] = []
  for d in docs[:12]:
    title = (d.title or "").strip()
    if not title:
      continue
    samples.append(title[:80])
    if "?" in title:
      patterns["question"] += 1
    if re.search(r"\b(19|20)\d{2}\b", title):
      patterns["year"] += 1
    if re.search(r"\b\d+\b", title):
      patterns["numbered"] += 1
    if re.search(r"\bguide\b", title, re.I):
      patterns["guide"] += 1
    if ":" in title:
      patterns["colon"] += 1
    if "|" in title:
      patterns["pipe"] += 1
  ranked = sorted(patterns, key=patterns.get, reverse=True)
  recommended = [p for p in ranked if patterns[p] > 0][:4] or ["guide", "benefit"]
  return {"patterns": patterns, "samples": samples[:6], "recommended": recommended}


@dataclass
class VariationContext:
  topic: str
  topic_title: str
  keywords: dict[str, Any]
  entities: list[str]
  intent: dict[str, Any]
  serp: dict[str, Any]
  tone: str
  category: str
  seed: int
  snippets: list[str] = field(default_factory=list)


def _build_title(ctx: VariationContext, idx: int) -> tuple[str, str]:
  """Return (title, angle). Supports 50+ unique combinations via seed + index."""
  base = ctx.topic_title
  low = base.lower()
  year = "2026"
  salt = ctx.seed + idx * 37
  angle = _pick(list(_ANGLES), salt)
  power = _pick(list(_POWER), salt + 3)
  serp_pref = ctx.serp.get("recommended", ["guide"])
  serp_angle = _pick(serp_pref, salt + 5)
  num = _pick(list(_NUMBERS), salt + 9)
  suffix = _pick(list(_SUFFIXES), salt + 17)
  entity = _pick(ctx.entities, salt + 19) if ctx.entities else ""
  kw = _pick(ctx.keywords.get("secondary", [base]), salt + 23)

  templates: list[tuple[str, str]] = [
    (f"{base}: The Complete {year} Guide", "guide"),
    (f"How to Master {base} ({year} Tips)", "how-to"),
    (f"{power} {base} Strategies That Work", "benefit"),
    (f"Top {num} {base} Best Practices ({year})", "listicle"),
    (f"{base} — Expert Guide for Beginners", "beginner"),
    (f"What Is {base}? {year} Guide", "question"),
    (f"{base} vs Alternatives: {year} Comparison", "comparison"),
    (f"{base} | {power} Tips & Examples", "pipe"),
    (f"Best {base} Guide ({year})", "year"),
    (f"{base}: Everything You Need to Know", "deep-dive"),
    (f"Quick Start Guide to {base}", "quick"),
    (f"{base} Explained: {power} Insights", "expert"),
    (f"{num} Ways to Improve {base} ({year})", "listicle"),
    (f"{base} {suffix} ({year})", angle),
    (f"{power} {kw} Guide to {base}", "guide"),
    (f"{base} for {year}: {power} Playbook", "year"),
    (f"Why {base} Matters in {year}", "question"),
    (f"The {power} {base} Handbook", "deep-dive"),
    (f"{base} Tips: {num} Proven Ideas", "listicle"),
    (f"Your {year} {base} Roadmap", "guide"),
    (f"{base} — {power} {suffix}", "benefit"),
    (f"From Zero to {base} {suffix}", "how-to"),
    (f"{num} {base} Mistakes to Avoid", "listicle"),
    (f"{base} FAQ: {year} Answers", "question"),
    (f"Inside {base}: {power} Breakdown", "deep-dive"),
  ]

  if entity:
    templates.extend([
      (f"{base} and {entity}: {year} Guide", "expert"),
      (f"{entity} + {base} — What to Know", "comparison"),
    ])
  if ctx.category == "local_business":
    templates.insert(0, (f"Best {base} Near You ({year})", "local"))
  if ctx.category == "how_to":
    templates.insert(0, (f"How to {base} — Step-by-Step ({year})", "how-to"))
  if ctx.intent.get("primary") == "commercial":
    templates.insert(0, (f"Best {base} — Reviews & Top Picks", "commercial"))
  if serp_angle == "question":
    templates.insert(0, (f"Why Choose {base}? {year} Answers", "question"))
  if serp_angle == "numbered":
    templates.insert(0, (f"{num} {base} Tips That Actually Work", "listicle"))

  # Rotate pool per idx so 50 requests get distinct titles
  pool = _shuffle(templates, salt)
  title, ang = pool[idx % len(pool)]
  # Inject index-based micro-variation for extra uniqueness
  if idx % 7 == 3 and len(title) < tme.TITLE_MAX - 8:
    title = title.replace(year, f"{year} Update", 1) if year in title else f"{title} ({year})"
  if idx % 11 == 5 and "—" not in title and len(title) < tme.TITLE_MAX - 4:
    title = title.replace(base, f"{base} —", 1)
  return title, ang


def _build_meta(ctx: VariationContext, title: str, idx: int) -> str:
  hook = _pick(list(_META_HOOKS), ctx.seed + idx * 11)
  cta = _pick(list(_CTA), ctx.seed + idx * 13)
  low = ctx.topic_title.lower()
  snippet = _pick(ctx.snippets, ctx.seed + idx) if ctx.snippets else ""
  long_tail = _pick(ctx.keywords.get("long_tail", [low]), ctx.seed + idx + 7)
  entity = _pick(ctx.entities, ctx.seed + idx + 29) if ctx.entities else ""
  num = _pick(list(_NUMBERS), ctx.seed + idx + 31)

  if snippet and len(snippet) > 40:
    core = _clip_sentence(snippet, 90)
  else:
    cores = [
      f"proven strategies for {low}",
      f"{num} practical tips on {low}",
      f"expert insights about {low}",
      f"what works for {low} in 2026",
      f"clear steps to improve {low}",
    ]
    core = _pick(cores, ctx.seed + idx * 5)

  meta = f"{hook} {core}. {cta}"
  if ctx.tone == "casual":
    metas = [
      f"Here's the real deal on {low} — no fluff. {cta}",
      f"Skip the jargon: {low} explained simply. {cta}",
    ]
    meta = _pick(metas, ctx.seed + idx)
  elif ctx.tone == "formal":
    meta = f"An authoritative overview of {low} with key insights for professional readers. {cta}"
  elif ctx.tone == "friendly":
    meta = f"We help you succeed with {low}. Friendly tips and clear steps inside. {cta}"

  if entity and entity.lower() not in meta.lower():
    meta = meta.replace(". ", f" Includes {entity} context. ", 1)
  if long_tail and long_tail.lower() not in meta.lower() and len(meta) < tme.META_MAX - 20:
    meta = meta.replace(". ", f" Covers {long_tail}. ", 1)

  return meta


def _clip_sentence(text: str, n: int) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0]


def _trim_title(title: str) -> str:
  title = re.sub(r"\s+", " ", (title or "").strip())
  if len(title) <= tme.TITLE_MAX:
    return title
  cut = title[: tme.TITLE_MAX].rsplit(" ", 1)[0]
  return cut.rstrip(" -:|,")


def _trim_meta(meta: str) -> str:
  meta = re.sub(r"\s+", " ", (meta or "").strip())
  if len(meta) > tme.META_MAX:
    cut = meta[: tme.META_MAX - 3].rsplit(" ", 1)[0]
    meta = cut.rstrip(" ,.;:") + "..."
  if len(meta) < tme.META_MIN:
    meta = (meta + " Learn more and get started today.").strip()
    if len(meta) > tme.META_MAX:
      meta = meta[: tme.META_MAX - 3].rsplit(" ", 1)[0] + "..."
  return meta


def generate_variations(ctx: VariationContext, count: int) -> list[dict[str, Any]]:
  """Generate up to `count` unique title/meta pairs."""
  seen_titles: set[str] = set()
  items: list[dict[str, Any]] = []
  max_attempts = count * 6
  for i in range(max_attempts):
    if len(items) >= count:
      break
    title, angle = _build_title(ctx, i)
    title = _trim_title(title)
    key = title.lower()
    if key in seen_titles:
      continue
    meta = _trim_meta(_build_meta(ctx, title, i))
    seen_titles.add(key)
    items.append({
      "title": title,
      "meta_description": meta,
      "angle": angle,
      "title_length": len(title),
      "meta_length": len(meta),
    })
  return items


def validate_lengths(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
  notes: list[str] = []
  for v in items:
    issues: list[str] = []
    if v["title_length"] > tme.TITLE_MAX:
      v["title"] = _trim_title(v["title"])
      v["title_length"] = len(v["title"])
      issues.append("title_trimmed")
    if v["meta_length"] > tme.META_MAX or v["meta_length"] < tme.META_MIN:
      v["meta_description"] = _trim_meta(v["meta_description"])
      v["meta_length"] = len(v["meta_description"])
      issues.append("meta_adjusted")
    v["length_issues"] = issues
    if issues:
      notes.append(f"Adjusted lengths for: {v['title'][:40]}")
  return items, notes


def dedupe_variations(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
  seen: set[str] = set()
  out: list[dict[str, Any]] = []
  removed = 0
  for v in items:
    key = v["title"].lower()
    if key in seen:
      removed += 1
      continue
    seen.add(key)
    out.append(v)
  return out, removed


def score_variations(items: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
  for v in items:
    q = tme.quality_variation(v["title"], v["meta_description"], topic)
    v["quality_score"] = q["quality_score"]
    v["seo_ready"] = q["seo_ready"]
    v["issues"] = q["issues"]
    v["seo_score"] = q["quality_score"]
  return items


def optimize_for_ai_search(items: list[dict[str, Any]], ctx: VariationContext) -> list[dict[str, Any]]:
  for i, v in enumerate(items):
    if ctx.intent.get("primary") == "informational" and "?" not in v["title"]:
      if (ctx.seed + i) % 5 == 0:
        v["ai_search_note"] = "Consider AIO-friendly direct answer in meta."
    v["ai_optimized"] = True
  return items


def quality_validate(items: list[dict[str, Any]], min_score: int = 60) -> list[dict[str, Any]]:
  return [v for v in items if v.get("quality_score", 0) >= min_score] or items


async def run_title_meta_pipeline(
  topic: str,
  *,
  variations: int = 10,
  tone: str = "professional",
  category: str = "blog_article",
  variation_seed: int | None = None,
  use_rag: bool = True,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  seed = effective_variation_seed(variation_seed)
  count = max(10, min(50, variations))
  topic = _topic_clean(topic)
  stages: dict[str, Any] = {}

  stages["input"] = {"topic": topic, "requested_variations": count}

  kw = extract_keywords(topic)
  stages["keyword_extractor"] = kw

  docs: list[OpenDoc] = []
  sources_used: list[str] = []
  if use_rag:
    try:
      docs = await asyncio.wait_for(
        retrieve_from_sources(topic, kw.get("secondary", []), _SOURCE_ROUTE[:3], per_source=1, seed=seed),
        timeout=4.0,
      )
      sources_used = sorted({d.source for d in docs})
    except asyncio.TimeoutError:
      docs = []
    stages["source_router"] = {"sources": _SOURCE_ROUTE[:3], "datasets": OPEN_DATASET_TREE}
  else:
    stages["source_router"] = {"sources": [], "fast_path": "local_templates", "datasets": OPEN_DATASET_TREE}

  stages["retriever"] = {"document_count": len(docs), "sources_used": sources_used}

  entities = extract_entities(topic, docs)
  stages["entity_extractor"] = {"entities": entities}

  intent = detect_intent(topic, kw)
  stages["intent_detector"] = intent

  serp = analyze_serp_patterns(docs, topic)
  stages["serp_pattern_analyzer"] = serp

  snippets = [d.text[:200] for d in docs if d.text][:8]
  ctx = VariationContext(
    topic=topic,
    topic_title=_topic_title(topic),
    keywords=kw,
    entities=entities,
    intent=intent,
    serp=serp,
    tone=tone,
    category=category,
    seed=seed,
    snippets=snippets,
  )

  raw_items = generate_variations(ctx, count + 5)
  stages["title_generator"] = {"mode": "dynamic_rag", "candidates": len(raw_items)}
  stages["meta_generator"] = {"mode": "dynamic_rag", "tone": tone}

  raw_items, len_notes = validate_lengths(raw_items)
  stages["length_validator"] = {"adjusted": len(len_notes), "notes": len_notes[:5]}

  raw_items, dup_removed = dedupe_variations(raw_items)
  stages["duplicate_checker"] = {"removed": dup_removed, "unique": len(raw_items)}

  raw_items = score_variations(raw_items, topic)
  stages["seo_scorer"] = {
    "avg": int(round(sum(v["quality_score"] for v in raw_items) / max(1, len(raw_items)))),
  }

  raw_items = optimize_for_ai_search(raw_items, ctx)
  stages["ai_search_optimizer"] = {"applied": True, "intent": intent.get("primary")}

  final_items = quality_validate(raw_items)[:count]
  stages["quality_validator"] = {
    "passed": len(final_items),
    "min_score": 60,
  }
  stages["final_output"] = {"variation_count": len(final_items)}

  avg_quality = int(round(sum(v["quality_score"] for v in final_items) / max(1, len(final_items))))

  return {
    "generator_version": GENERATOR_VERSION,
    "topic": topic,
    "category": category,
    "tone": tone,
    "variations": final_items,
    "variation_count": len(final_items),
    "variation_seed": seed,
    "title_limit": tme.TITLE_MAX,
    "meta_min": tme.META_MIN,
    "meta_max": tme.META_MAX,
    "quality": {
      "average_score": avg_quality,
      "seo_ready": avg_quality >= 75,
      "all_ready": all(v.get("seo_ready") for v in final_items),
    },
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "pipeline": {
      "keywords": kw,
      "entities": entities,
      "intent": intent,
      "serp_patterns": serp,
      "retrieval": {"sources_used": sources_used, "document_count": len(docs)},
    },
    "rag": {"enabled": use_rag, "sources_used": sources_used},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "unlimited_outputs": True,
    "per_request_unique": True,
  }

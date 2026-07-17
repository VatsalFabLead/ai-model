"""Content-first SEO Keyword Strategist — hosted-LLM JSON generation.

Follows topical authority / intent / entity rules. Never invents industries
or entities not present in the provided title/content.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.services.provider_base import ModelProvider

STRATEGIST_VERSION = "seo-keyword-strategist-v1"

_SYSTEM = """You are an expert SEO strategist with deep knowledge of Google Search, topical authority, semantic SEO, keyword clustering, and search intent analysis.

Your task is to generate highly relevant SEO keywords for the provided content.

IMPORTANT RULES
1. Generate keywords ONLY related to the given title and content.
2. Never hallucinate industries, companies, locations, products, or entities.
3. Do NOT generate generic AI keywords unless they directly match the topic.
4. If the article is a blog, prioritize informational keywords.
5. If the article is a product page, prioritize commercial keywords.
6. If the article is a service page, prioritize transactional keywords.
7. Local SEO keywords should ONLY be generated when the content is location-dependent.
8. Do not generate keywords like "near me", "company", "agency", "services", "providers" unless the content actually discusses those topics.
9. Every keyword must be semantically related to the article.
10. Remove duplicate or overly similar keywords.

Return JSON only. No markdown fences. No commentary.
Use this exact schema:
{
  "classification": {
    "content_type": "Blog|Tutorial|News|Landing Page|Product Page|Service Page|Documentation|Comparison",
    "search_intent": "Informational|Commercial Investigation|Transactional|Navigational",
    "industry": "string from content only"
  },
  "entities": {
    "products": [],
    "technologies": [],
    "concepts": [],
    "standards": [],
    "frameworks": [],
    "companies": [],
    "protocols": [],
    "security_terms": []
  },
  "topics": ["5-15 topics ranked by relevance"],
  "keyword_groups": {
    "primary_keyword": "string",
    "secondary_keywords": ["up to 10"],
    "long_tail_keywords": ["up to 20"],
    "question_keywords": ["up to 10"],
    "lsi_keywords": ["up to 20"],
    "trending_variations": ["up to 10"],
    "commercial_keywords": ["only if applicable, else []"],
    "local_keywords": ["only if location evidence in content, else []"],
    "competitor_keywords": ["only if applicable, else []"],
    "opportunity_keywords": ["high-opportunity phrases"]
  },
  "keyword_metrics": [
    {
      "keyword": "string",
      "search_volume": "Very Low|Low|Medium|High",
      "keyword_difficulty": "Easy|Medium|Hard",
      "competition": "Low|Medium|High",
      "opportunity_score": 0
    }
  ],
  "clusters": {
    "Cluster Name": ["keyword1", "keyword2"]
  },
  "validation": {
    "rejected_examples": [],
    "kept_count": 0
  }
}

Include every unique keyword in keyword_metrics. opportunity_score is 0-100 (qualitative AI estimate only).
"""

_VOLUME_MAP = {
  "very low": "very_low",
  "very_low": "very_low",
  "low": "low",
  "medium": "medium",
  "high": "high",
  "very high": "very_high",
  "very_high": "very_high",
}
_DIFF_MAP = {
  "easy": "low",
  "low": "low",
  "medium": "medium",
  "hard": "high",
  "high": "high",
}
_COMP_MAP = {"low": "low", "medium": "medium", "high": "high"}
_VOLUME_LABELS = {
  "very_low": "Very Low",
  "low": "Low",
  "medium": "Medium",
  "high": "High",
  "very_high": "Very High",
}
_VOLUME_RANGES = {
  "very_low": "<100",
  "low": "100–1K",
  "medium": "1K–10K",
  "high": "10K–100K",
  "very_high": "100K+",
}
_LEVEL_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}
_INTENT_MAP = {
  "informational": "informational",
  "commercial investigation": "commercial",
  "commercial": "commercial",
  "transactional": "transactional",
  "navigational": "navigational",
}

_GENERIC_BANNED = (
  "near me", " hire ", "agency", " providers", "development company",
  "software development company", "ai development services",
)


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _extract_json(raw: str) -> dict[str, Any]:
  text = (raw or "").strip()
  if not text:
    raise ValueError("empty strategist response")
  fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
  if fence:
    text = fence.group(1).strip()
  start = text.find("{")
  end = text.rfind("}")
  if start < 0 or end <= start:
    raise ValueError("no JSON object in strategist response")
  return json.loads(text[start : end + 1])


def _norm_kw(kw: str) -> str:
  return _clean(kw).lower().strip(" .,;:\"'")


_FOREIGN_LEAK = (
  "healthcare", "telemedicine", "hipaa", "hospital management", "medical software",
  "machine learning development", "ai development services", "hire flutter",
  "software development company", "iot healthcare", "patient monitoring",
)


def _is_banned_generic(kw: str, content_low: str) -> bool:
  k = f" {_norm_kw(kw)} "
  if "near me" in k and "near me" not in content_low and not re.search(
    r"\b(city|town|location|local|in [a-z]{3,}|pune|mumbai|delhi|india|usa|london)\b",
    content_low,
  ):
    return True
  for ban in ("agency", "providers", "hire developers", "software development company"):
    if ban in k and ban not in content_low:
      return True
  if re.search(r"\b(company|services)\b", k) and not re.search(r"\b(company|services|service)\b", content_low):
    if "company" in k and "company" not in content_low:
      return True
    if "services" in k and not re.search(r"\bservices?\b", content_low):
      return True
  for leak in _FOREIGN_LEAK:
    if leak in k and leak not in content_low:
      return True
  return False


def _semantic_ok(kw: str, anchors: set[str], content_low: str) -> bool:
  toks = {t for t in re.findall(r"[a-z0-9]+", kw.lower()) if len(t) > 2}
  if not toks:
    return False
  if not anchors:
    return True
  if toks & anchors:
    return True
  # Prefix soft-match (roast/roasting, coffee/coffees)
  for t in toks:
    for a in anchors:
      if len(t) >= 4 and len(a) >= 4 and (t[:4] == a[:4]):
        return True
  # Trust strategist LSI unless clearly foreign to the article
  for leak in _FOREIGN_LEAK:
    if leak in kw and leak not in content_low:
      return False
  return True


def _anchor_tokens(*texts: str) -> set[str]:
  stop = {
    "the", "and", "for", "with", "from", "that", "this", "your", "our", "are",
    "was", "have", "will", "can", "into", "about", "their", "what", "when",
    "where", "which", "how", "why", "best", "guide", "using", "into",
  }
  out: set[str] = set()
  for text in texts:
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
      if len(t) > 2 and t not in stop:
        out.add(t)
  return out


def build_strategist_prompt(
  *,
  title: str,
  content: str,
  primary_topic: str,
  country: str,
  language: str,
) -> str:
  return (
    "INPUT\n\n"
    f"Title:\n{title or '(not provided)'}\n\n"
    f"Content:\n{content}\n\n"
    f"Primary Topic:\n{primary_topic or '(infer from content)'}\n\n"
    f"Market:\n{country or 'Global'}\n\n"
    f"Language:\n{language or 'English'}\n\n"
    "Follow STEP 1–7 from your instructions. Return JSON only."
  )


def _collect_group_keywords(groups: dict[str, Any]) -> list[tuple[str, str]]:
  """Return (keyword, category) pairs in priority order."""
  order = [
    ("primary_keyword", "primary"),
    ("secondary_keywords", "secondary"),
    ("long_tail_keywords", "long_tail"),
    ("question_keywords", "questions"),
    ("lsi_keywords", "lsi"),
    ("trending_variations", "trending"),
    ("commercial_keywords", "commercial"),
    ("local_keywords", "local"),
    ("competitor_keywords", "commercial"),
    ("opportunity_keywords", "opportunity"),
  ]
  out: list[tuple[str, str]] = []
  seen: set[str] = set()
  for key, cat in order:
    val = groups.get(key)
    items: list[str]
    if isinstance(val, str):
      items = [val]
    elif isinstance(val, list):
      items = [str(x) for x in val]
    else:
      items = []
    for raw in items:
      kw = _norm_kw(raw)
      if not kw or len(kw) < 3 or len(kw) > 90 or kw in seen:
        continue
      seen.add(kw)
      out.append((kw, cat))
  return out


def _metrics_index(metrics: list[Any]) -> dict[str, dict[str, Any]]:
  idx: dict[str, dict[str, Any]] = {}
  for m in metrics or []:
    if not isinstance(m, dict):
      continue
    kw = _norm_kw(str(m.get("keyword") or ""))
    if kw:
      idx[kw] = m
  return idx


def _cluster_for(kw: str, clusters: dict[str, Any]) -> str:
  low = kw.lower()
  for name, members in (clusters or {}).items():
    if not isinstance(members, list):
      continue
    for m in members:
      if _norm_kw(str(m)) == low:
        return str(name)
  return "General"


def _intent_for(kw: str, default_intent: str, category: str) -> str:
  k = kw.lower()
  if category == "questions" or k.split()[0] in ("what", "how", "why", "when", "where", "which", "who"):
    return "informational"
  if category == "commercial" or any(x in k for x in ("buy", "price", "pricing", "cost", "subscription")):
    return "commercial"
  if category == "local" or "near me" in k:
    return "transactional"
  return default_intent


def strategist_payload_to_result(
  data: dict[str, Any],
  *,
  title: str,
  content: str,
  primary_topic: str,
  country: str,
  language: str,
  max_keywords: int = 50,
  elapsed_ms: float = 0,
  backend: str | None = None,
) -> dict[str, Any]:
  content_low = content.lower()
  anchors = _anchor_tokens(title, content, primary_topic)
  classification = data.get("classification") or {}
  entities = data.get("entities") or {}
  topics = [str(t) for t in (data.get("topics") or []) if str(t).strip()][:15]
  groups = data.get("keyword_groups") or {}
  clusters_raw = data.get("clusters") or {}
  metrics_idx = _metrics_index(data.get("keyword_metrics") or [])

  intent_raw = str(classification.get("search_intent") or "Informational").lower()
  default_intent = _INTENT_MAP.get(intent_raw, "informational")
  content_type = str(classification.get("content_type") or "Blog")
  industry = str(classification.get("industry") or primary_topic or "General")

  primary = _norm_kw(str(groups.get("primary_keyword") or primary_topic or title or "topic"))
  seed = primary or _norm_kw(primary_topic) or _norm_kw(title)[:80] or "topic"

  flat = _collect_group_keywords(groups)
  # Ensure primary is first
  if primary and not any(k == primary for k, _ in flat):
    flat.insert(0, (primary, "primary"))

  rejected: list[str] = []
  kept_pairs: list[tuple[str, str]] = []
  for kw, cat in flat:
    if _is_banned_generic(kw, content_low):
      rejected.append(kw)
      continue
    if not _semantic_ok(kw, anchors, content_low):
      rejected.append(kw)
      continue
    kept_pairs.append((kw, cat))

  rows: list[dict[str, Any]] = []
  for kw, cat in kept_pairs:
    m = metrics_idx.get(kw, {})
    vol = _VOLUME_MAP.get(str(m.get("search_volume") or "medium").strip().lower(), "medium")
    diff = _DIFF_MAP.get(str(m.get("keyword_difficulty") or "medium").strip().lower(), "medium")
    comp = _COMP_MAP.get(str(m.get("competition") or "medium").strip().lower(), "medium")
    try:
      opp = int(m.get("opportunity_score", 70))
    except (TypeError, ValueError):
      opp = 70
    opp = max(0, min(100, opp))
    intent = _intent_for(kw, default_intent, cat)
    cluster = _cluster_for(kw, clusters_raw)
    relevance = 95 if cat == "primary" else 88 if cat == "secondary" else 78
    rows.append({
      "keyword": kw,
      "category": cat if cat != "opportunity" else "secondary",
      "topic_cluster": cluster,
      "is_competitor": cat == "commercial" and "competitor" in str(groups.get("competitor_keywords") or "").lower(),
      "is_trending": cat == "trending",
      "volume_estimate": vol,
      "volume_label": _VOLUME_LABELS[vol],
      "volume_range": _VOLUME_RANGES[vol],
      "difficulty_estimate": diff,
      "difficulty_label": _LEVEL_LABELS[diff],
      "cpc_estimate": "medium" if intent == "commercial" else "low",
      "cpc_label": "Medium" if intent == "commercial" else "Low",
      "cpc_range": "$1–3" if intent == "commercial" else "$0–1",
      "competition_estimate": comp,
      "competition_label": _LEVEL_LABELS[comp],
      "trend": "up" if cat == "trending" else "stable",
      "trend_icon": "📈" if cat == "trending" else "➜",
      "trend_monthly": [],
      "trend_chart": "",
      "intent": intent,
      "relevance_score": relevance,
      "topic_relevance": relevance,
      "sources": ["seo_strategist", "hosted_llm"],
      "metrics_source": "ai_estimate",
      "seo_score": min(100, int(opp * 0.6 + relevance * 0.4)),
      "opportunity_score": opp,
      "opportunity_breakdown": {
        "volume": _VOLUME_LABELS[vol],
        "difficulty": _LEVEL_LABELS[diff],
        "competition": _LEVEL_LABELS[comp],
        "intent": intent,
      },
    })

  # Prefer diversity then opportunity
  rows.sort(key=lambda r: (r["opportunity_score"], r["relevance_score"]), reverse=True)
  # Keep primary first if present
  primary_rows = [r for r in rows if r["keyword"] == primary]
  other = [r for r in rows if r["keyword"] != primary]
  ranked = (primary_rows + other)[: max(10, min(50, max_keywords))]

  by_cat: dict[str, list[dict[str, Any]]] = {}
  for r in ranked:
    by_cat.setdefault(r["category"], []).append(r)

  topic_clusters: dict[str, list[str]] = {}
  for name, members in (clusters_raw or {}).items():
    if isinstance(members, list):
      cleaned = [_norm_kw(str(m)) for m in members if _norm_kw(str(m))]
      cleaned = [c for c in cleaned if any(r["keyword"] == c for r in ranked)]
      if cleaned:
        topic_clusters[str(name)] = cleaned

  output = {
    "primary_keywords": by_cat.get("primary", [])[:5],
    "secondary_keywords": by_cat.get("secondary", [])[:15],
    "long_tail_keywords": by_cat.get("long_tail", [])[:20],
    "question_keywords": by_cat.get("questions", [])[:10],
    "lsi_keywords": by_cat.get("lsi", [])[:20],
    "trending_keywords": [r for r in ranked if r.get("is_trending")][:10],
    "commercial_keywords": by_cat.get("commercial", [])[:15],
    "local_keywords": by_cat.get("local", [])[:10],
    "competitor_keywords": [],
    "opportunity_keywords": sorted(ranked, key=lambda x: x["opportunity_score"], reverse=True)[:12],
    "entities": entities,
    "topics": topics,
    "seo_score": {"overall": int(sum(r["seo_score"] for r in ranked) / max(len(ranked), 1))},
  }

  opportunities = [
    r for r in ranked
    if r["opportunity_score"] >= 70 and r["difficulty_estimate"] in ("low", "medium")
  ][:10]

  backend_name = backend or "hosted"
  return {
    "generator_version": STRATEGIST_VERSION,
    "seed_keyword": seed,
    "context": content[:2000] if content else None,
    "title": title or None,
    "primary_topic": primary_topic or seed,
    "market": country or None,
    "language": language or "English",
    "count": len(ranked),
    "keywords": ranked,
    "keyword_categories": by_cat,
    "topic_clusters": topic_clusters,
    "clusters": [
      {"name": name, "keywords": kws, "size": len(kws)}
      for name, kws in topic_clusters.items()
    ],
    "opportunities": opportunities,
    "output": output,
    "summary": {
      "content_type": content_type,
      "search_intent": classification.get("search_intent"),
      "industry": industry,
      "topic_count": len(topics),
      "rejected_off_topic": len(rejected),
      "metrics_source": "ai_estimate",
    },
    "seo_score": output["seo_score"],
    "recommendations": [
      f"Primary keyword focus: {seed}",
      f"Content type: {content_type} → lean {default_intent}",
      f"Industry (from content): {industry}",
    ][:5],
    "metrics_source": "ai_estimate",
    "metrics_disclaimer": (
      "Search volume, CPC, difficulty, and competition are AI estimates — not data from "
      "Google Ads, Search Console, Ahrefs, or Semrush. Use qualitative labels for planning only."
    ),
    "discovery": {"enabled": False, "sources_used": ["hosted_llm_strategist"], "queries_run": 0, "errors": []},
    "architecture": {
      "flow": [
        "content_input", "classification", "entity_extraction", "topic_extraction",
        "keyword_groups", "metrics_estimation", "clustering", "validation", "final_output",
      ],
      "stages": {
        "classification": classification,
        "entity_extraction": entities,
        "topic_extraction": {"topics": topics, "count": len(topics)},
        "validation": {
          "rejected": rejected[:20],
          "rejected_count": len(rejected),
          "kept": len(ranked),
        },
      },
      "mode": "content_strategist",
    },
    "pipeline": {
      "context": {
        "title": title,
        "primary_topic": primary_topic,
        "market": country,
        "language": language,
        "content_type": content_type,
        "industry": industry,
      },
      "entities": [
        e for group in entities.values() if isinstance(group, list) for e in group
      ][:30],
      "primary_domain": industry,
      "seed_intent": {"primary_intent": default_intent},
      "topics": topics,
    },
    "rag": {"enabled": False, "sources_used": []},
    "elapsed_ms": elapsed_ms,
    "ai": {
      "enabled": True,
      "model_used": True,
      "backend": backend_name,
      "hosted": True,
      "mode": "strategist",
    },
    "strategist": data,
  }


async def generate_with_strategist(
  provider: ModelProvider,
  *,
  title: str = "",
  content: str = "",
  primary_topic: str = "",
  country: str = "",
  language: str = "English",
  max_keywords: int = 50,
) -> dict[str, Any]:
  title = _clean(title)
  content = (content or "").strip()
  primary_topic = _clean(primary_topic)
  country = _clean(country)
  language = _clean(language) or "English"

  if not content and not title and not primary_topic:
    raise ValueError("Provide title and/or content (or primary_topic) for strategist mode")

  if not content:
    content = " ".join(x for x in (title, primary_topic) if x)

  if not title and primary_topic:
    title = primary_topic
  if not primary_topic and title:
    primary_topic = title

  t0 = time.perf_counter()
  user_prompt = build_strategist_prompt(
    title=title,
    content=content[:8000],
    primary_topic=primary_topic,
    country=country,
    language=language,
  )
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=_SYSTEM,
    use_rag=False,
    skip_intent=True,
    max_tokens=3500,
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
    title=title,
    content=content,
    primary_topic=primary_topic,
    country=country,
    language=language,
    max_keywords=max_keywords,
    elapsed_ms=elapsed,
    backend=str(backend) if backend else "hosted",
  )

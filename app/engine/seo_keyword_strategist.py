"""Content-first SEO Keyword Strategist v3 — search-like keywords + SERP/gap analysis.

Relevance over volume. Similarity is a hard validation gate (not display-only).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from app.services.provider_base import ModelProvider

STRATEGIST_VERSION = "seo-keyword-strategist-v3"
MIN_SEMANTIC_SIMILARITY = 80

_SYSTEM = """You are an elite SEO Strategist with expertise in Google Search, semantic SEO, topical authority, keyword research, search intent analysis, SERP analysis, and content-gap strategy.

Your objective is to generate highly relevant, SEARCH-LIKE SEO keywords that real users would type into Google — not broad technical labels.

## Core Principles
1. Prioritize relevance over search volume.
2. Every keyword must directly relate to the article's primary topic.
3. Never generate keywords from unrelated industries.
4. Never hallucinate entities, brands, companies, locations, or products.
5. Avoid generic keywords unless they accurately describe the article.
6. Think like an SEO strategist, not a keyword expander.
7. Generate keywords that users would realistically search to find this content.
8. Prefer topical authority over keyword stuffing.
9. Diversify keyword lengths (short, medium, long-tail).
10. Reject weak semantic matches (semantic_similarity must be >= 80).

## Search-like keyword style (CRITICAL)
Write queries the way people search — natural language, not catalog terms.

GOOD (search-like):
- how to prioritize vulnerabilities with AI
- AI vulnerability prioritization examples
- risk based vulnerability management guide
- what is AI threat detection

BAD (broad / technical labels):
- Artificial Intelligence
- Vulnerability Management
- Cybersecurity
- Machine Learning
- AI Technology

Prefer 3–8 word phrases. Include modifiers: how, what, why, guide, examples, best practices, checklist, vs, for beginners, techniques, tools (only if content supports tools).

## Quality Rules
Each keyword MUST include:
- intent (Informational | Commercial Investigation | Transactional | Navigational)
- reason (1 clear sentence explaining why it fits THIS article)
- semantic_similarity (0–100) — used for VALIDATION; below 80 = reject
- confidence_score (0–100)
- unique metric estimates (do NOT reuse the same volume/difficulty/competition/opportunity across many keywords)

Reject: "AI company", "AI services", "AI agency", "AI near me" unless the article is about companies/services.
Local keywords ONLY if location-specific. Commercial/competitor ONLY if products/services are discussed.

## Metric variety
Distribute estimates realistically across Very Low/Low/Medium/High and Easy/Medium/Hard.
Opportunity scores must vary (spread across roughly 45–95). Never assign identical metrics to every keyword.

## Content gaps & SERP
Also return:
- content_gap_analysis: topics/queries the article should cover but currently under-covers
- serp_predictions: likely SERP features / ranking notes for the primary and top secondary keywords

Return JSON only. No markdown fences. No commentary.
Schema:
{
  "classification": {
    "main_topic": "string",
    "secondary_topics": ["string"],
    "user_intent": "string",
    "target_audience": "string",
    "content_type": "Blog|Tutorial|News|Landing Page|Product Page|Service Page|Documentation|Comparison",
    "search_intent": "Informational|Commercial Investigation|Transactional|Navigational",
    "industry": "string from content only",
    "confidence": {
      "main_topic": 0,
      "content_type": 0,
      "search_intent": 0,
      "industry": 0,
      "target_audience": 0
    }
  },
  "entities": {
    "technologies": [],
    "frameworks": [],
    "programming_languages": [],
    "standards": [],
    "security_concepts": [],
    "products": [],
    "companies": [],
    "ai_models": [],
    "cloud_platforms": [],
    "protocols": []
  },
  "topic_graph": {
    "primary_topic": "string",
    "supporting_topics": ["string"],
    "related_concepts": ["string"],
    "synonyms": ["string"],
    "semantic_relationships": [{"from": "string", "to": "string", "relation": "string"}]
  },
  "topics": ["5-15 topics ranked by relevance"],
  "keyword_groups": {
    "primary_keyword": "string",
    "secondary_keywords": ["up to 10"],
    "long_tail_keywords": ["up to 20"],
    "question_keywords": ["up to 10"],
    "lsi_keywords": ["up to 20"],
    "trending_variations": ["up to 10"],
    "commercial_keywords": [],
    "local_keywords": [],
    "competitor_keywords": [],
    "opportunity_keywords": []
  },
  "keywords": [
    {
      "keyword": "string",
      "intent": "Informational|Commercial Investigation|Transactional|Navigational",
      "search_volume": "Very Low|Low|Medium|High",
      "keyword_difficulty": "Easy|Medium|Hard",
      "competition": "Low|Medium|High",
      "opportunity_score": 0,
      "semantic_similarity": 0,
      "confidence_score": 0,
      "reason": "1 sentence why this keyword is relevant to THIS article",
      "category": "primary|secondary|long_tail|questions|lsi|trending|commercial|local|competitor|opportunity"
    }
  ],
  "content_gap_analysis": {
    "covered_well": ["topics the article already covers"],
    "missing_topics": ["important related topics not covered"],
    "underdeveloped_angles": ["angles mentioned weakly that need depth"],
    "recommended_sections": ["H2/H3 ideas to close gaps"],
    "priority_gap_keywords": ["search queries that expose the gaps"]
  },
  "serp_predictions": [
    {
      "keyword": "string",
      "likely_serp_features": ["People Also Ask", "Featured Snippet", "Videos", "Images", "Top Stories", "Local Pack", "Discussions"],
      "ranking_difficulty_note": "1 sentence",
      "content_format_to_win": "guide|listicle|comparison|howto|definition|tool page",
      "predicted_ctr_opportunity": "Low|Medium|High"
    }
  ],
  "clusters": {"Cluster Name": ["keyword1", "keyword2"]},
  "recommendations": ["string"],
  "validation": {
    "rejected_examples": [{"keyword": "string", "reason": "string", "semantic_similarity": 0}],
    "kept_count": 0,
    "min_similarity": 80
  }
}

Include EVERY kept keyword in "keywords" with unique realistic metrics and a non-empty reason + intent.
Remove any keyword with semantic_similarity below 80 before returning.
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
_VOL_CYCLE = ("very_low", "low", "medium", "high", "low", "medium", "very_low", "medium", "high", "low")
_DIFF_CYCLE = ("low", "medium", "high", "medium", "low", "medium", "high", "low", "medium", "medium")
_COMP_CYCLE = ("low", "medium", "high", "medium", "low", "high", "medium", "low", "medium", "high")

_FOREIGN_LEAK = (
  "healthcare", "telemedicine", "hipaa", "hospital management", "medical software",
  "machine learning development", "ai development services", "hire flutter",
  "software development company", "iot healthcare", "patient monitoring",
  "ai company", "ai services", "ai agency", "ai near me",
)

_BROAD_TECH_ONLY = frozenset({
  "artificial intelligence", "machine learning", "deep learning", "cybersecurity",
  "information technology", "cloud computing", "data science", "big data",
  "software", "technology", "ai", "ml", "nlp", "iot", "blockchain",
  "vulnerability management", "threat detection", "security", "automation",
})


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


def _is_broad_technical_label(kw: str) -> bool:
  k = _norm_kw(kw)
  if k in _BROAD_TECH_ONLY:
    return True
  words = k.split()
  if len(words) <= 2 and not any(
    m in k for m in ("how", "what", "why", "guide", "examples", "best", "for", "with", "vs")
  ):
    if words and words[0] in ("ai", "ml", "seo") and len(words) == 1:
      return True
  return False


def _is_banned_generic(kw: str, content_low: str, *, informational: bool) -> bool:
  k = f" {_norm_kw(kw)} "
  banned_ai = ("ai company", "ai services", "ai agency", "ai near me", "best ai", "artificial intelligence company")
  for ban in banned_ai:
    if ban in k and ban not in content_low and "company" not in content_low and "agency" not in content_low:
      return True
  if "near me" in k and "near me" not in content_low and not re.search(
    r"\b(city|town|location|local|in [a-z]{3,}|pune|mumbai|delhi|india|usa|london)\b",
    content_low,
  ):
    return True
  for ban in ("agency", "providers", "hire developers", "software development company"):
    if ban in k and ban not in content_low:
      return True
  if informational and re.search(r"\b(company|services|agency|providers|hire)\b", k):
    if not re.search(r"\b(company|services|agency|service|hire|provider)\b", content_low):
      return True
  for leak in _FOREIGN_LEAK:
    if leak in k and leak not in content_low:
      return True
  if _is_broad_technical_label(kw):
    return True
  return False


def _anchor_tokens(*texts: str) -> set[str]:
  stop = {
    "the", "and", "for", "with", "from", "that", "this", "your", "our", "are",
    "was", "have", "will", "can", "into", "about", "their", "what", "when",
    "where", "which", "how", "why", "best", "guide", "using",
  }
  out: set[str] = set()
  for text in texts:
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
      if len(t) > 2 and t not in stop:
        out.add(t)
  return out


def _local_similarity(kw: str, anchors: set[str], content_low: str) -> int:
  """Deterministic similarity used as a validation signal (0–100)."""
  toks = {t for t in re.findall(r"[a-z0-9]+", kw.lower()) if len(t) > 2}
  if not toks:
    return 0
  if not anchors:
    return 70
  overlap = toks & anchors
  score = 0
  if overlap:
    score += min(55, 18 * len(overlap))
  for t in toks:
    for a in anchors:
      if len(t) >= 4 and len(a) >= 4 and t[:4] == a[:4]:
        score += 12
        break
  if kw.lower() in content_low:
    score += 25
  elif any(t in content_low for t in toks if len(t) >= 4):
    score += 15
  if len(kw.split()) >= 3:
    score += 8
  if len(kw.split()) >= 5:
    score += 4
  if _is_broad_technical_label(kw):
    score = min(score, 55)
  for leak in _FOREIGN_LEAK:
    if leak in kw and leak not in content_low:
      return min(score, 40)
  return max(0, min(100, score))


def _semantic_ok(kw: str, anchors: set[str], content_low: str) -> bool:
  return _local_similarity(kw, anchors, content_low) >= 55


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
    f"Primary Keyword:\n{primary_topic or '(infer from content)'}\n\n"
    f"Language:\n{language or 'English'}\n\n"
    f"Target Market:\n{country or 'Global'}\n\n"
    "Generate search-like keywords (how people actually Google), varied metrics, "
    "per-keyword intent + reason, content_gap_analysis, and serp_predictions. "
    f"VALIDATION: drop any keyword with semantic_similarity < {MIN_SEMANTIC_SIMILARITY}. "
    "Return JSON only."
  )


def _collect_group_keywords(groups: dict[str, Any]) -> list[tuple[str, str]]:
  order = [
    ("primary_keyword", "primary"),
    ("secondary_keywords", "secondary"),
    ("long_tail_keywords", "long_tail"),
    ("question_keywords", "questions"),
    ("lsi_keywords", "lsi"),
    ("trending_variations", "trending"),
    ("commercial_keywords", "commercial"),
    ("local_keywords", "local"),
    ("competitor_keywords", "competitor"),
    ("opportunity_keywords", "opportunity"),
  ]
  out: list[tuple[str, str]] = []
  seen: set[str] = set()
  for key, cat in order:
    val = groups.get(key)
    if isinstance(val, str):
      items = [val]
    elif isinstance(val, list):
      items = [str(x) for x in val]
    else:
      items = []
    for raw in items:
      if isinstance(raw, dict):
        raw = str(raw.get("keyword") or "")
      kw = _norm_kw(str(raw))
      if not kw or len(kw) < 3 or len(kw) > 90 or kw in seen:
        continue
      seen.add(kw)
      out.append((kw, cat))
  return out


def _safe_int(val: Any, default: int) -> int:
  try:
    return int(val)
  except (TypeError, ValueError):
    return default


def _cluster_for(kw: str, clusters: dict[str, Any]) -> str:
  low = kw.lower()
  for name, members in (clusters or {}).items():
    if not isinstance(members, list):
      continue
    for m in members:
      item = m.get("keyword") if isinstance(m, dict) else m
      if _norm_kw(str(item)) == low:
        return str(name)
  return "General"


def _intent_label(raw: str, default: str = "informational") -> str:
  return _INTENT_MAP.get((raw or "").strip().lower(), default)


def _intent_from_phrase(kw: str, default: str) -> str:
  k = kw.lower()
  if k.split() and k.split()[0] in ("what", "how", "why", "when", "where", "which", "who"):
    return "informational"
  if any(x in k for x in ("buy", "pricing", "price", "cost", "subscribe", "order", "download")):
    return "transactional" if any(x in k for x in ("buy", "order", "subscribe", "download")) else "commercial"
  if any(x in k for x in ("best", "vs", "versus", "review", "compare", "alternative")):
    return "commercial"
  if "near me" in k or re.search(r"\bin [a-z]{3,}\b", k):
    return "transactional"
  return default


def _hash_idx(kw: str, mod: int) -> int:
  return int(hashlib.md5(kw.encode("utf-8")).hexdigest()[:8], 16) % mod


def _diversify_metrics(
  vol: str,
  diff: str,
  comp: str,
  opp: int,
  kw: str,
  *,
  force: bool,
) -> tuple[str, str, str, int]:
  if not force:
    jitter = _hash_idx(kw, 7) - 3
    opp = max(42, min(96, opp + jitter))
    return vol, diff, comp, opp
  i = _hash_idx(kw, 10)
  words = len(kw.split())
  vol = _VOL_CYCLE[i]
  diff = _DIFF_CYCLE[i]
  comp = _COMP_CYCLE[i]
  if words >= 5:
    vol = "very_low" if i % 2 == 0 else "low"
    diff = "low" if i % 3 else "medium"
  elif words <= 2:
    vol = "high" if i % 2 == 0 else "medium"
    diff = "high" if i % 3 == 0 else "medium"
  opp = 48 + _hash_idx(kw + ":opp", 48)
  return vol, diff, comp, opp


def _metrics_are_collapsed(rows_preview: list[dict[str, Any]]) -> bool:
  if len(rows_preview) < 4:
    return False
  vols = {r.get("search_volume") or r.get("volume_estimate") for r in rows_preview}
  diffs = {r.get("keyword_difficulty") or r.get("difficulty_estimate") for r in rows_preview}
  opps = {r.get("opportunity_score") for r in rows_preview}
  return len(vols) <= 1 or len(diffs) <= 1 or len(opps) <= 2


def _normalize_keyword_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
  entries: dict[str, dict[str, Any]] = {}
  for item in data.get("keywords") or []:
    if not isinstance(item, dict):
      continue
    kw = _norm_kw(str(item.get("keyword") or ""))
    if not kw:
      continue
    entries[kw] = dict(item)
    entries[kw]["keyword"] = kw
    if not entries[kw].get("category"):
      entries[kw]["category"] = "secondary"

  for kw, cat in _collect_group_keywords(data.get("keyword_groups") or {}):
    if kw not in entries:
      entries[kw] = {"keyword": kw, "category": cat}
    else:
      entries[kw].setdefault("category", cat)

  primary = _norm_kw(str((data.get("keyword_groups") or {}).get("primary_keyword") or ""))
  if primary and primary not in entries:
    entries[primary] = {"keyword": primary, "category": "primary"}
  elif primary and primary in entries:
    entries[primary]["category"] = "primary"

  return list(entries.values())


def _normalize_content_gaps(raw: Any) -> dict[str, list[str]]:
  if not isinstance(raw, dict):
    return {
      "covered_well": [],
      "missing_topics": [],
      "underdeveloped_angles": [],
      "recommended_sections": [],
      "priority_gap_keywords": [],
    }
  out: dict[str, list[str]] = {}
  for key in (
    "covered_well", "missing_topics", "underdeveloped_angles",
    "recommended_sections", "priority_gap_keywords",
  ):
    vals = raw.get(key) or []
    if isinstance(vals, list):
      out[key] = [_clean(str(v)) for v in vals if _clean(str(v))][:12]
    else:
      out[key] = []
  return out


def _normalize_serp(raw: Any, allowed: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  if not isinstance(raw, list):
    return out
  for item in raw:
    if not isinstance(item, dict):
      continue
    kw = _norm_kw(str(item.get("keyword") or ""))
    if not kw:
      continue
    _ = allowed  # keep signature for callers filtering later
    features = item.get("likely_serp_features") or []
    if not isinstance(features, list):
      features = []
    out.append({
      "keyword": kw,
      "likely_serp_features": [str(f) for f in features[:8]],
      "ranking_difficulty_note": _clean(str(item.get("ranking_difficulty_note") or ""))[:240],
      "content_format_to_win": _clean(str(item.get("content_format_to_win") or "guide"))[:40],
      "predicted_ctr_opportunity": _clean(str(item.get("predicted_ctr_opportunity") or "Medium")),
    })
    if len(out) >= 12:
      break
  return out


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
  topic_graph = data.get("topic_graph") or {}
  topics = [str(t) for t in (data.get("topics") or []) if str(t).strip()][:15]
  if not topics and topic_graph.get("supporting_topics"):
    topics = [str(t) for t in topic_graph.get("supporting_topics") or []][:15]
  groups = data.get("keyword_groups") or {}
  clusters_raw = data.get("clusters") or {}
  llm_recs = [str(r) for r in (data.get("recommendations") or []) if str(r).strip()]
  content_gaps = _normalize_content_gaps(data.get("content_gap_analysis"))

  intent_raw = str(classification.get("search_intent") or "Informational")
  default_intent = _intent_label(intent_raw)
  informational = default_intent == "informational"
  content_type = str(classification.get("content_type") or "Blog")
  industry = str(
    classification.get("industry")
    or classification.get("main_topic")
    or topic_graph.get("primary_topic")
    or primary_topic
    or "General"
  )
  main_topic = str(
    classification.get("main_topic")
    or topic_graph.get("primary_topic")
    or primary_topic
    or title
    or "topic"
  )

  primary = _norm_kw(str(groups.get("primary_keyword") or primary_topic or main_topic or title or "topic"))
  seed = primary or _norm_kw(primary_topic) or _norm_kw(title)[:80] or "topic"

  rejected: list[dict[str, Any]] = []
  for ex in (data.get("validation") or {}).get("rejected_examples") or []:
    if isinstance(ex, dict):
      rejected.append({
        "keyword": str(ex.get("keyword") or ""),
        "reason": str(ex.get("reason") or ""),
        "semantic_similarity": _safe_int(ex.get("semantic_similarity"), 0),
        "validation_stage": "model",
      })
    elif ex:
      rejected.append({
        "keyword": str(ex),
        "reason": "rejected by model",
        "semantic_similarity": 0,
        "validation_stage": "model",
      })

  entries = _normalize_keyword_entries(data)
  collapse = _metrics_are_collapsed(entries)

  validation_checks = {
    "min_similarity": MIN_SEMANTIC_SIMILARITY,
    "require_intent": True,
    "require_reason": True,
    "reject_broad_technical_labels": True,
    "reject_generic_agency_phrases": True,
    "similarity_is_gate": True,
  }

  rows: list[dict[str, Any]] = []
  seen: set[str] = set()
  for item in entries:
    kw = _norm_kw(str(item.get("keyword") or ""))
    if not kw or kw in seen:
      continue
    seen.add(kw)
    cat = str(item.get("category") or "secondary").lower().replace(" ", "_")
    if cat == "trending_variations":
      cat = "trending"
    if cat not in {
      "primary", "secondary", "long_tail", "questions", "lsi", "trending",
      "commercial", "local", "competitor", "opportunity",
    }:
      cat = "secondary"

    local_sim = _local_similarity(kw, anchors, content_low)
    model_sim = _safe_int(item.get("semantic_similarity"), 0)
    if model_sim <= 0:
      sim = local_sim
    else:
      sim = int(0.55 * model_sim + 0.45 * local_sim)
      if local_sim < 50:
        sim = min(sim, local_sim + 20)

    reject_reason = ""
    if sim < MIN_SEMANTIC_SIMILARITY:
      reject_reason = f"validation failed: semantic_similarity {sim} < {MIN_SEMANTIC_SIMILARITY}"
    elif _is_banned_generic(kw, content_low, informational=informational):
      reject_reason = "validation failed: broad/generic or agency-style phrase"
    elif not _semantic_ok(kw, anchors, content_low):
      reject_reason = "validation failed: weak semantic match to title/content"
    elif informational and cat in ("commercial", "competitor"):
      if not re.search(r"\b(buy|price|product|service|pricing|purchase|shop)\b", content_low):
        reject_reason = "validation failed: commercial keyword on informational content"
    elif cat == "local" and not re.search(
      r"\b(city|town|location|local|near me|pune|mumbai|delhi|india|usa|london)\b",
      content_low,
    ):
      reject_reason = "validation failed: local keyword without location evidence"

    if reject_reason:
      rejected.append({
        "keyword": kw,
        "reason": reject_reason,
        "semantic_similarity": sim,
        "model_similarity": model_sim,
        "local_similarity": local_sim,
        "validation_stage": "similarity_gate",
      })
      continue

    vol = _VOLUME_MAP.get(str(item.get("search_volume") or item.get("volume") or "medium").strip().lower(), "medium")
    diff = _DIFF_MAP.get(
      str(item.get("keyword_difficulty") or item.get("difficulty") or "medium").strip().lower(),
      "medium",
    )
    comp = _COMP_MAP.get(str(item.get("competition") or "medium").strip().lower(), "medium")
    opp = max(0, min(100, _safe_int(item.get("opportunity_score") or item.get("opportunity"), 70)))
    vol, diff, comp, opp = _diversify_metrics(vol, diff, comp, opp, kw, force=collapse)

    conf = max(0, min(100, _safe_int(item.get("confidence_score") or item.get("confidence"), sim)))
    intent_raw_kw = str(item.get("intent") or "").strip()
    intent = _intent_label(intent_raw_kw, "") if intent_raw_kw else ""
    if not intent:
      intent = _intent_from_phrase(kw, default_intent)
    if cat == "questions":
      intent = "informational"

    reason = _clean(str(item.get("reason") or ""))
    if not reason:
      reason = (
        f"Matches how readers search for {main_topic} and aligns with the article's "
        f"{'informational' if informational else content_type.lower()} focus."
      )

    cluster = _cluster_for(kw, clusters_raw)
    relevance = max(sim, 95 if cat == "primary" else 0)

    rows.append({
      "keyword": kw,
      "category": "secondary" if cat in ("opportunity", "competitor") else cat,
      "topic_cluster": cluster,
      "is_competitor": cat == "competitor",
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
      "topic_relevance": sim,
      "semantic_similarity": sim,
      "model_similarity": model_sim,
      "local_similarity": local_sim,
      "confidence_score": conf,
      "reason": reason,
      "sources": ["seo_strategist_v3", "hosted_llm"],
      "metrics_source": "ai_estimate",
      "seo_score": min(100, int(0.45 * sim + 0.35 * opp + 0.20 * conf)),
      "opportunity_score": opp,
      "opportunity_breakdown": {
        "volume": _VOLUME_LABELS[vol],
        "difficulty": _LEVEL_LABELS[diff],
        "competition": _LEVEL_LABELS[comp],
        "intent": intent,
        "semantic_similarity": sim,
      },
      "validation": {
        "passed": True,
        "similarity_gate": MIN_SEMANTIC_SIMILARITY,
        "similarity_score": sim,
      },
    })

  vol_rank = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
  intent_match = {"informational": 4, "commercial": 3, "transactional": 2, "navigational": 1}

  def _sort_key(r: dict[str, Any]) -> tuple:
    intent_fit = 4 if r["intent"] == default_intent else intent_match.get(r["intent"], 1)
    long_tail = 1 if len(r["keyword"].split()) >= 4 else 0
    search_like = 1 if len(r["keyword"].split()) >= 3 else 0
    return (
      r.get("semantic_similarity", 0),
      intent_fit,
      search_like,
      r.get("opportunity_score", 0),
      vol_rank.get(r.get("volume_estimate", "low"), 1),
      long_tail,
      r.get("confidence_score", 0),
    )

  rows.sort(key=_sort_key, reverse=True)
  primary_rows = [r for r in rows if r["keyword"] == primary]
  other = [r for r in rows if r["keyword"] != primary]
  ranked = (primary_rows + other)[: max(10, min(50, max_keywords))]
  kept_set = {r["keyword"] for r in ranked}

  by_cat: dict[str, list[dict[str, Any]]] = {}
  for r in ranked:
    by_cat.setdefault(r["category"], []).append(r)

  topic_clusters: dict[str, list[str]] = {}
  for name, members in (clusters_raw or {}).items():
    if not isinstance(members, list):
      continue
    cleaned: list[str] = []
    for m in members:
      item = m.get("keyword") if isinstance(m, dict) else m
      c = _norm_kw(str(item))
      if c and c in kept_set:
        cleaned.append(c)
    if cleaned:
      topic_clusters[str(name)] = cleaned

  serp = _normalize_serp(data.get("serp_predictions"), kept_set | {seed, primary})

  output = {
    "primary_keywords": by_cat.get("primary", [])[:5] or primary_rows[:1],
    "secondary_keywords": by_cat.get("secondary", [])[:15],
    "long_tail_keywords": by_cat.get("long_tail", [])[:20],
    "question_keywords": by_cat.get("questions", [])[:10],
    "lsi_keywords": by_cat.get("lsi", [])[:20],
    "trending_keywords": [r for r in ranked if r.get("is_trending")][:10],
    "commercial_keywords": by_cat.get("commercial", [])[:15],
    "local_keywords": by_cat.get("local", [])[:10],
    "competitor_keywords": [r for r in ranked if r.get("is_competitor")][:10],
    "opportunity_keywords": sorted(ranked, key=lambda x: x["opportunity_score"], reverse=True)[:12],
    "entities": entities,
    "topics": topics,
    "topic_graph": topic_graph,
    "content_gap_analysis": content_gaps,
    "serp_predictions": serp,
    "seo_score": {"overall": int(sum(r["seo_score"] for r in ranked) / max(len(ranked), 1))},
  }

  opportunities = [
    r for r in ranked
    if r["opportunity_score"] >= 70 and r["difficulty_estimate"] in ("low", "medium")
  ][:10]

  conf = classification.get("confidence") if isinstance(classification.get("confidence"), dict) else {}
  gap_recs = [
    f"Close content gap: {g}" for g in (content_gaps.get("missing_topics") or [])[:3]
  ]
  recommendations = (llm_recs[:5] + gap_recs)[:8] or [
    f"Primary keyword focus: {seed}",
    f"Content type: {content_type} → lean {default_intent}",
    f"Main topic: {main_topic} (confidence {conf.get('main_topic', '—')})",
  ]

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
    "content_gap_analysis": content_gaps,
    "serp_predictions": serp,
    "output": output,
    "summary": {
      "content_type": content_type,
      "search_intent": classification.get("search_intent"),
      "industry": industry,
      "main_topic": main_topic,
      "target_audience": classification.get("target_audience"),
      "confidence": conf,
      "topic_count": len(topics),
      "rejected_off_topic": len(rejected),
      "min_semantic_similarity": MIN_SEMANTIC_SIMILARITY,
      "similarity_is_validation_gate": True,
      "metrics_diversified": collapse,
      "content_gaps": len(content_gaps.get("missing_topics") or []),
      "serp_predictions": len(serp),
      "metrics_source": "ai_estimate",
    },
    "seo_score": output["seo_score"],
    "recommendations": recommendations,
    "metrics_source": "ai_estimate",
    "metrics_disclaimer": (
      "Search volume, CPC, difficulty, and competition are AI estimates — not data from "
      "Google Ads, Search Console, Ahrefs, or Semrush. Use qualitative labels for planning only."
    ),
    "discovery": {"enabled": False, "sources_used": ["hosted_llm_strategist_v3"], "queries_run": 0, "errors": []},
    "architecture": {
      "flow": [
        "content_input", "classification", "entity_extraction", "topic_graph",
        "keyword_groups", "intent_and_explanations", "metrics_estimation",
        "content_gap_analysis", "serp_predictions", "clustering",
        "similarity_validation_gate", "prioritization", "final_output",
      ],
      "stages": {
        "classification": classification,
        "entity_extraction": entities,
        "topic_graph": topic_graph,
        "topic_extraction": {"topics": topics, "count": len(topics)},
        "content_gap_analysis": content_gaps,
        "serp_predictions": {"count": len(serp), "items": serp[:5]},
        "validation": {
          "checks": validation_checks,
          "rejected": rejected[:25],
          "rejected_count": len(rejected),
          "kept": len(ranked),
          "min_similarity": MIN_SEMANTIC_SIMILARITY,
          "similarity_is_gate": True,
          "metrics_diversified": collapse,
        },
      },
      "mode": "content_strategist_v3",
    },
    "pipeline": {
      "context": {
        "title": title,
        "primary_topic": primary_topic,
        "market": country,
        "language": language,
        "content_type": content_type,
        "industry": industry,
        "main_topic": main_topic,
      },
      "entities": [
        e for group in entities.values() if isinstance(group, list) for e in group
      ][:40],
      "primary_domain": industry,
      "seed_intent": {"primary_intent": default_intent},
      "topics": topics,
      "topic_graph": topic_graph,
      "content_gap_analysis": content_gaps,
      "serp_predictions": serp,
    },
    "rag": {"enabled": False, "sources_used": []},
    "elapsed_ms": elapsed_ms,
    "ai": {
      "enabled": True,
      "model_used": True,
      "backend": backend_name,
      "hosted": True,
      "mode": "strategist_v3",
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
    max_tokens=5000,
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

"""Seed/context-anchored relevance for SEO keyword retrieval & ranking."""

from __future__ import annotations

import re
from typing import Any

_STOP = frozenset({
  "the", "and", "for", "with", "from", "that", "this", "your", "our", "are",
  "was", "were", "have", "has", "had", "will", "can", "into", "onto", "about",
  "their", "them", "they", "what", "when", "where", "which", "who", "how",
  "why", "best", "top", "near", "india", "company", "services", "service",
  "solutions", "solution", "providers", "provider", "online", "guide", "tips",
  "cost", "pricing", "hire", "buy", "vs", "versus", "step", "beginners",
  "professional", "affordable", "official", "using", "across", "selling",
  "offering", "provide", "provides", "run", "running", "boutique", "home",
})

# Leak terms that must not appear unless the seed/domain is in that vertical
_TECH_HEALTH_LEAK = (
  "healthcare", "telemedicine", "hipaa", "hospital management", "medical software",
  "medical ai", "clinical", "patient monitoring", "ehr", "emr",
  "flutter app", "machine learning", "software development", "app development company",
  "hire developers", "hire flutter", "ai development", "iot healthcare",
  "firebase", "python medical", "computer vision healthcare", "radiology",
  "digital transformation services", "it solutions company", "custom software",
  "ai chatbot for healthcare", "healthcare software", "medical app",
)

_CATEGORY_LEAK_BLOCKS: dict[str, tuple[str, ...]] = {
  "Food": _TECH_HEALTH_LEAK + ("saas", "crm development", "erp development"),
  "Beauty & Fashion": _TECH_HEALTH_LEAK,
  "Travel": _TECH_HEALTH_LEAK,
  "Real Estate & Property": _TECH_HEALTH_LEAK,
  "Automotive": _TECH_HEALTH_LEAK,
  "Education": ("hipaa", "telemedicine", "escort"),
  "Local Business": _TECH_HEALTH_LEAK,
}

_HEALTHCARE_DOMAINS = frozenset({
  "Healthcare", "Telemedicine", "Hospital", "Diagnostics", "Pharmacy",
  "Mental Health", "Dental", "Nutrition",
})
_TECH_DOMAINS = frozenset({
  "Artificial Intelligence", "Machine Learning", "Mobile App Development",
  "Web Development", "Software", "Cloud Computing", "IoT", "Firebase",
  "Flutter", "Python", "Computer Vision", "SEO", "Technology",
})


def _tokens(text: str) -> set[str]:
  return {
    t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(t) > 2 and t not in _STOP
  }


def anchor_tokens(context: dict[str, Any]) -> set[str]:
  """Tokens that define the user's actual topic."""
  parts = [
    str(context.get("seed_keyword") or ""),
    str(context.get("normalized_seed") or ""),
    str(context.get("context_brief") or "")[:800],
    str(context.get("primary_domain") or ""),
    str(context.get("brand_name") or ""),
    " ".join(str(t) for t in (context.get("extracted_topics") or [])[:8]),
    " ".join(str(c) for c in (context.get("topic_clusters") or [])[:6]),
  ]
  toks = _tokens(" ".join(parts))
  domain = str(context.get("primary_domain") or "").lower()
  for piece in domain.replace("&", " ").split():
    if len(piece) > 2:
      toks.add(piece)
  return toks


def expansion_domains(context: dict[str, Any]) -> list[str]:
  """Only expand from primary (+ tightly related) domains — cuts cross-vertical noise."""
  primary = context.get("primary_domain") or "Business"
  scores = context.get("domain_scores") or {}
  listed = list(context.get("domains") or [primary])
  if primary not in listed:
    listed = [primary] + listed

  top = int(scores.get(primary, 3) or 3)
  category = context.get("domain_category") or ""
  tight = category in ("Food", "Beauty & Fashion", "Travel", "Local Business", "Real Estate & Property")

  kept: list[str] = []
  for d in listed:
    if d == primary:
      kept.append(d)
      continue
    sc = int(scores.get(d, 0) or 0)
    # Require strong secondary signal; never pull tech/health into food/etc.
    if tight and d in (_HEALTHCARE_DOMAINS | _TECH_DOMAINS):
      continue
    if sc >= max(3, top - 1) or (not tight and sc >= max(2, top - 2)):
      kept.append(d)
    if len(kept) >= (2 if tight else 3):
      break
  return kept or [primary]


def topic_template_allowed(cluster: str, context: dict[str, Any]) -> bool:
  """Legacy topic packs only when they match the user's domain/seed."""
  primary = (context.get("primary_domain") or "").lower()
  cluster_l = (cluster or "").lower()
  if not cluster_l:
    return False
  if cluster_l == primary:
    return True
  brief = " ".join([
    str(context.get("seed_keyword") or ""),
    str(context.get("context_brief") or ""),
  ]).lower()
  if cluster_l in brief or cluster_l.replace(" ", "") in brief.replace(" ", ""):
    return True
  # Never apply healthcare/AI packs to unrelated verticals
  if cluster in _HEALTHCARE_DOMAINS | {"Artificial Intelligence", "Flutter", "Python", "Computer Vision", "Machine Learning"}:
    if (context.get("primary_domain") or "") not in _HEALTHCARE_DOMAINS | _TECH_DOMAINS:
      return False
  return False


def has_category_leak(keyword: str, context: dict[str, Any]) -> bool:
  k = keyword.lower()
  category = context.get("domain_category") or ""
  primary = context.get("primary_domain") or ""

  blocked = _CATEGORY_LEAK_BLOCKS.get(category, ())
  if any(b in k for b in blocked):
    return True

  # Healthcare jargon outside healthcare vertical
  if primary not in _HEALTHCARE_DOMAINS and any(
    b in k for b in ("healthcare", "hipaa", "telemedicine", "patient monitoring", "medical diagnosis")
  ):
    return True

  # Generic AI-agency spam outside tech
  if primary not in _TECH_DOMAINS and any(
    b in k for b in (
      "hire ai developers", "ai development services", "ai software development company",
      "custom ai solutions", "machine learning development company",
    )
  ):
    return True
  return False


def topic_relevance_score(keyword: str, context: dict[str, Any]) -> int:
  """0–100 relevance of a candidate to the user's seed/context."""
  k = (keyword or "").lower().strip()
  if not k:
    return 0
  if has_category_leak(k, context):
    return 0
  # Drop pronoun/filler garbage from bad phrase splits
  if re.match(r"^(we|our|i|my)\b", k) or " we " in f" {k} ":
    return 0

  anchors = anchor_tokens(context)
  kw_toks = _tokens(k)
  if not kw_toks:
    return 0

  overlap = kw_toks & anchors
  domain = (context.get("primary_domain") or "").lower()
  seed = str(context.get("seed_keyword") or context.get("normalized_seed") or "").lower()
  score = 0
  if seed and (seed in k or k in seed):
    score += 55
  if domain and domain in k:
    score += 40
  if overlap:
    score += min(50, 20 * len(overlap))
  # Soft credit for domain stem in keyword (Cafe → cafe)
  for a in list(anchors)[:40]:
    if len(a) >= 4 and a in k:
      score += 8
      break

  # Penalize purely generic agency phrasing with no topic overlap
  generic = ("services", "company", "providers", "solutions", "agency", "hire")
  if any(g in k for g in generic) and len(overlap) < 2 and (not seed or seed.split()[0] not in k):
    score -= 30

  return max(0, min(100, score))


def passes_topic_relevance(keyword: str, context: dict[str, Any], *, min_score: int = 28) -> bool:
  return topic_relevance_score(keyword, context) >= min_score

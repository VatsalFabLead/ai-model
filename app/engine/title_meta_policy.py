"""Title & Meta — policy gate, escort disambiguation, safe fallback copy."""

from __future__ import annotations

import re
from typing import Any

from app.engine.seo_retrieval_engine import (
  detect_content_profile,
  detect_nsfw_topic,
  disambiguate_entities_embedding,
  entity_candidates,
)
from app.engine.title_meta_enrichment import trim_meta, trim_title

TITLE_MAX = 60
META_MIN = 140
META_MAX = 160

# Promotional / explicit patterns — never allow in titles or metas
_POLICY_BLOCKED_PATTERNS: tuple[str, ...] = (
  r"\bbest\s+\w+\s+escort",
  r"\bmaster\s+.+\s+escort",
  r"\bhow\s+to\s+.+\s+escort",
  r"\bescort\s+strateg",
  r"\bescort\s+tips\b",
  r"\bcall\s+girl",
  r"\bxxx\b",
  r"\bporn",
  r"\bnude\b",
  r"\bnsfw\b",
  r"\berotic\b",
  r"\bhookup\b",
  r"\bsugar\s+baby\b",
)

_SAFE_TITLE_SETS: dict[str, list[tuple[str, str]]] = {
  "adult_services": [
    ("Companion Services: Safety & Privacy Guide ({year})", "safety_guide"),
    ("Escort Regulations: What You Should Know ({year})", "regulations"),
    ("Companion Services Explained — Key Facts ({year})", "informational"),
    ("Privacy & Safety in Companion Services ({year})", "privacy"),
    ("Understanding Escort Service Policies ({year})", "policies"),
    ("Companion Industry: Legal & Safety Overview ({year})", "legal_overview"),
  ],
  "ambiguous_escort": [
    ("Escort Explained: Security, Auto & Services ({year})", "disambiguation"),
    ("What Does Escort Mean? A Clear Guide ({year})", "definition"),
    ("Escort Terminology: Security vs Automotive ({year})", "terminology"),
    ("Understanding 'Escort' in Different Contexts ({year})", "contexts"),
  ],
  "automotive": [
    ("{topic_title}: Buyer's Guide & Tips ({year})", "buyers_guide"),
    ("{topic_title} — Specs, History & Maintenance ({year})", "maintenance"),
    ("Used {topic_title}: What to Check Before Buying ({year})", "used_guide"),
    ("{topic_title}: Complete Owner Overview ({year})", "owner_guide"),
  ],
  "security_escort": [
    ("{topic_title}: Planning & Protocol Guide ({year})", "protocol"),
    ("VIP Security Escort Services Explained ({year})", "vip_guide"),
    ("{topic_title} — Risk Assessment Basics ({year})", "risk"),
    ("Security Escort Procedures: Overview ({year})", "procedures"),
  ],
  "local_services": [
    ("{topic_title}: Local Provider Guide ({year})", "local_guide"),
    ("{topic_title} Near You — What to Know ({year})", "local_near"),
    ("Choosing a {topic_title} Provider: Key Tips ({year})", "choosing"),
  ],
}

_SAFE_META: dict[str, tuple[str, ...]] = {
  "adult_services": (
    "Informational overview of companion service safety, privacy practices, and regulatory "
    "considerations. For research and awareness — not promotional content.",
    "Learn about privacy, consent, and safety standards relevant to companion services. "
    "Educational resource with practical compliance-oriented guidance.",
    "A neutral guide covering policies, discretion, and safety expectations in the companion "
    "services industry. Read before making any decisions.",
  ),
  "ambiguous_escort": (
    "Clarifies how the term escort is used across security, automotive, and service contexts. "
    "Educational disambiguation — not promotional material.",
    "Explains different meanings of escort including VIP security details and vehicle model names. "
    "Helps readers find the right information.",
  ),
  "automotive": (
    "Practical buyer and owner information covering specs, reliability, maintenance, and value "
    "factors. Compare options and make an informed decision.",
    "Overview of features, history, and maintenance tips for informed car buyers and owners.",
  ),
  "security_escort": (
    "Professional overview of security escort planning, protocols, and risk assessment for "
    "events, convoys, and VIP movements.",
    "Learn when security escorts are required and how teams coordinate safe transport.",
  ),
  "local_services": (
    "Local provider guide covering how to evaluate options, pricing factors, and service "
    "expectations in your area.",
  ),
}


def _clean(text: str) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _topic_title(topic: str) -> str:
  t = _clean(topic)
  return " ".join(w.capitalize() for w in t.split()) if t else "Your Topic"


def analyze_topic_policy(topic: str, keywords: dict[str, Any] | None = None) -> dict[str, Any]:
  """Pre-generation policy analysis — profile, disambiguation, RAG gates."""
  kw = keywords or {}
  secondary = list(kw.get("secondary") or [])
  long_tail = list(kw.get("long_tail") or [])[:4]
  keyword_list = secondary + long_tail

  nsfw = detect_nsfw_topic(topic, keyword_list)
  profile = nsfw["profile"]
  candidates = entity_candidates(topic, keyword_list)
  disambiguation = disambiguate_entities_embedding(topic, keyword_list, candidates)

  escort_profiles = ("adult_services", "ambiguous_escort", "automotive", "security_escort")
  restricted = profile in ("adult_services", "ambiguous_escort")
  use_safe_only = profile in escort_profiles or nsfw.get("use_template_only", False)

  return {
    "profile": profile,
    "is_adult": nsfw.get("is_adult", False),
    "is_ambiguous": nsfw.get("is_ambiguous", False),
    "policy_status": (
      "restricted_informational" if restricted else
      "specialized_safe" if profile in ("automotive", "security_escort") else
      "allowed"
    ),
    "skip_open_retrieval": nsfw.get("skip_open_retrieval", False) or profile in escort_profiles,
    "use_safe_templates": use_safe_only,
    "block_promotional": restricted or profile in ("automotive", "security_escort"),
    "entity_candidates": candidates,
    "disambiguation": disambiguation,
    "resolved_entity": disambiguation.get("selected"),
    "disambiguation_method": disambiguation.get("method"),
    "message": (
      "Restricted topic: generating informational titles only — no promotional adult SEO copy."
      if restricted else None
    ),
  }


def contains_blocked_copy(text: str) -> bool:
  low = text.lower()
  return any(re.search(p, low) for p in _POLICY_BLOCKED_PATTERNS)


def validate_policy_compliance(
  title: str,
  meta: str,
  policy: dict[str, Any],
) -> dict[str, Any]:
  """Post-generation policy check."""
  issues: list[str] = []
  if policy.get("block_promotional"):
    if contains_blocked_copy(title):
      issues.append("promotional_title_blocked")
    if contains_blocked_copy(meta):
      issues.append("promotional_meta_blocked")
    promo_words = ("best strategies", "master ", "step-by-step guide to", "proven strategies")
    if any(p in title.lower() for p in promo_words):
      issues.append("promotional_framing_blocked")
  return {"valid": not issues, "issues": issues}


def generate_safe_variations(
  topic: str,
  policy: dict[str, Any],
  *,
  count: int,
  seed: int,
  year: str = "2026",
) -> list[dict[str, Any]]:
  """Informational title/meta pairs for restricted or disambiguated topics."""
  profile = policy.get("profile", "general")
  if profile not in _SAFE_TITLE_SETS:
    profile = "ambiguous_escort" if "escort" in topic.lower() else "local_services"

  topic_title = _topic_title(topic)
  templates = list(_SAFE_TITLE_SETS.get(profile, _SAFE_TITLE_SETS["local_services"]))
  metas = _SAFE_META.get(profile, _SAFE_META["local_services"])

  seen: set[str] = set()
  items: list[dict[str, Any]] = []
  for i in range(count * 3):
    if len(items) >= count:
      break
    tpl, angle = templates[(seed + i) % len(templates)]
    title = tpl.format(topic_title=topic_title, year=year)
    title = trim_title(title)
    if title.lower() in seen or contains_blocked_copy(title):
      continue
    meta_raw = metas[(seed + i * 7) % len(metas)]
    meta = trim_meta(meta_raw)
    compliance = validate_policy_compliance(title, meta, policy)
    if not compliance["valid"]:
      continue
    seen.add(title.lower())
    items.append({
      "title": title,
      "meta_description": meta,
      "angle": angle,
      "title_length": len(title),
      "meta_length": len(meta),
      "validation_issues": [],
      "policy_safe": True,
      "overall_score": 88,
      "seo_score": 82,
      "ctr_score": 75,
      "quality_score": 88,
      "seo_ready": True,
      "issues": [],
    })
  return items[:count]


def filter_policy_compliant(
  items: list[dict[str, Any]],
  policy: dict[str, Any],
) -> list[dict[str, Any]]:
  """Remove variations that fail post-generation policy validation."""
  if not policy.get("block_promotional"):
    return items
  out: list[dict[str, Any]] = []
  for v in items:
    check = validate_policy_compliance(
      v.get("title", ""),
      v.get("meta_description", ""),
      policy,
    )
    if check["valid"]:
      v["policy_safe"] = True
      out.append(v)
    else:
      v["policy_safe"] = False
      v["issues"] = list(dict.fromkeys((v.get("issues") or []) + check["issues"]))
  return out

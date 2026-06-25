"""SEO Keyword Generator — pipeline stages, validation, and structured output."""

from __future__ import annotations

import re
from typing import Any

METRICS_SOURCE = "ai_estimate"

ARCHITECTURE_FLOW = [
  "input",
  "input_validator",
  "spell_correction",
  "language_detector",
  "country_detection",
  "named_entity_recognition",
  "entity_disambiguation",
  "brand_detection",
  "industry_classification",
  "search_intent_detection",
  "knowledge_graph_lookup",
  "domain_rule_engine",
  "open_data_retrieval",
  "keyword_expansion",
  "question_generator",
  "competitor_generator",
  "lsi_semantic_keywords",
  "local_seo_keywords",
  "trending_keywords",
  "volume_estimator",
  "difficulty_estimator",
  "cpc_estimator",
  "competition_estimator",
  "opportunity_scoring",
  "keyword_clustering",
  "ranking_prioritization",
  "seo_quality_validator",
  "final_output",
]

OPEN_DATASET_TREE: dict[str, list[str]] = {}  # populated from seo_keyword_open_data at import

def _load_open_dataset_tree() -> dict[str, list[str]]:
  from app.engine.seo_keyword_open_data import OPEN_DATASET_TREE as tree
  return tree

OPEN_DATASET_TREE = _load_open_dataset_tree()

_LANG_HINTS: dict[str, tuple[str, ...]] = {
  "en": ("english", "the", "and", "for", "with", "development", "software"),
  "hi": ("hindi", "भारत", "हिंदी"),
  "es": ("spanish", "español", "desarrollo"),
  "fr": ("french", "français", "développement"),
  "de": ("german", "deutsch", "entwicklung"),
  "pt": ("portuguese", "português"),
  "ar": ("arabic", "العربية"),
}

_COUNTRY_REGIONS: dict[str, tuple[str, ...]] = {
  "India": ("india", "gujarat", "surat", "ahmedabad", "mumbai", "delhi", "bangalore", "hyderabad", "chennai"),
  "United States": ("usa", "united states", "california", "texas", "new york"),
  "United Kingdom": ("uk", "united kingdom", "london", "england"),
  "Canada": ("canada", "toronto", "vancouver"),
  "Australia": ("australia", "sydney", "melbourne"),
  "UAE": ("uae", "dubai", "abu dhabi"),
  "Singapore": ("singapore",),
}

_INDUSTRY_HINTS: dict[str, tuple[str, ...]] = {
  "Healthcare": ("healthcare", "medical", "hospital", "patient", "hipaa", "telemedicine", "clinical", "ehr"),
  "Technology": ("software", "app", "development", "flutter", "python", "firebase", "cloud", "api"),
  "Artificial Intelligence": ("artificial intelligence", "machine learning", "ai", "computer vision", "nlp"),
  "Beauty": ("beauty", "cosmetic", "cosmetics", "makeup", "skincare", "lipstick", "sugar"),
  "Cosmetics": ("cosmetic", "cosmetics", "makeup", "beauty", "skincare", "lip", "foundation"),
  "Finance": ("fintech", "banking", "insurance", "payment", "trading"),
  "E-commerce": ("ecommerce", "e-commerce", "shop", "retail", "marketplace", "product", "company"),
  "Education": ("education", "learning", "course", "training", "edtech"),
  "Marketing": ("seo", "marketing", "advertising", "digital marketing"),
  "Food": ("food", "restaurant", "recipe", "organic", "grocery"),
}

_TYPO_FIXES = (
  (r"\bprouct\b", "product"),
  (r"\bcomapny\b", "company"),
  (r"\bcomapnies\b", "companies"),
  (r"\bbeauty\b", "beauty"),
)


def normalize_seed_typos(seed: str) -> str:
  out = seed
  for pattern, repl in _TYPO_FIXES:
    out = re.sub(pattern, repl, out, flags=re.I)
  return _clean(out)


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def validate_input(seed_keyword: str) -> dict[str, Any]:
  seed = _clean(seed_keyword)
  issues: list[str] = []
  if not seed:
    issues.append("seed_keyword_required")
  elif len(seed) < 2:
    issues.append("seed_keyword_too_short")
  return {
    "valid": not issues,
    "issues": issues,
    "seed_length": len(seed),
    "word_count": len(seed.split()),
  }


def detect_language(seed: str, requested: str | None = None) -> dict[str, Any]:
  low = seed.lower()
  if requested:
    code = requested.strip().lower()[:2]
    return {"language": requested, "bcp47": code if len(code) == 2 else "en", "source": "user"}
  scores: dict[str, int] = {}
  for code, hints in _LANG_HINTS.items():
    scores[code] = sum(1 for h in hints if h in low)
  best = max(scores, key=scores.get) if scores else "en"
  if scores.get(best, 0) == 0:
    best = "en"
  labels = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "ar": "Arabic"}
  return {"language": labels.get(best, "English"), "bcp47": best, "source": "auto_detect", "scores": scores}


def detect_country_region(seed: str, context: dict[str, Any]) -> dict[str, Any]:
  haystack = (seed + " " + " ".join(context.get("locations", []))).lower()
  detected: list[str] = []
  for country, hints in _COUNTRY_REGIONS.items():
    if any(h in haystack for h in hints):
      detected.append(country)
  if not detected:
    detected = ["Global"]
  return {
    "countries": detected,
    "regions": context.get("locations", []),
    "primary_market": detected[0],
  }


def recognize_brand_entity(context: dict[str, Any]) -> dict[str, Any]:
  return {
    "brand_name": context.get("brand_name"),
    "is_brand_seed": context.get("is_brand_seed", False),
    "topic_mode": context.get("topic_mode", False),
    "entities": [e.get("name") for e in context.get("entities", []) if isinstance(e, dict)],
  }


def classify_industry_domain(seed: str, context: dict[str, Any]) -> dict[str, Any]:
  """Domain-first classification (190 domains) with legacy industry shape."""
  from app.engine.seo_keyword_domains import classify_domains
  from app.engine.seo_keyword_domain_engine import classify_industry_from_domain

  domain_info = classify_domains(seed, context)
  return classify_industry_from_domain(domain_info)


def detect_seed_intent(seed: str, context: dict[str, Any]) -> dict[str, Any]:
  low = seed.lower()
  if any(w in low for w in ("hire", "buy", "pricing", "cost", "agency", "company")):
    primary = "commercial"
  elif any(w in low for w in ("how to", "what is", "why", "guide", "tutorial")):
    primary = "informational"
  elif any(w in low for w in ("near me", "book", "order")):
    primary = "transactional"
  elif context.get("is_brand_seed"):
    primary = "navigational"
  else:
    primary = "commercial"
  return {
    "primary_intent": primary,
    "intents": ["informational", "commercial", "transactional", "navigational"],
  }


def normalize_seed_keyword(seed: str, context: dict[str, Any]) -> dict[str, Any]:
  normalized = _clean(seed.lower())
  tokens = [t for t in re.findall(r"\w+", normalized) if len(t) > 1]
  core_phrases = context.get("topic_parts") or []
  return {
    "normalized_seed": normalized,
    "core_phrases": core_phrases[:12],
    "token_count": len(tokens),
    "topic_cluster_count": len(context.get("topic_clusters", [])),
  }


def expand_lsi_keywords(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  from app.engine.seo_keyword_domain_engine import generate_domain_lsi
  return generate_domain_lsi(context, existing)


def generate_question_keywords(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  from app.engine.seo_keyword_domain_engine import generate_domain_questions
  return generate_domain_questions(context, existing)


def generate_local_seo_keywords(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  from app.engine.seo_keyword_domain_engine import generate_domain_local
  return generate_domain_local(context, existing)


def generate_competitor_keywords(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  from app.engine.seo_keyword_domain_engine import generate_domain_competitors
  return generate_domain_competitors(context, existing)


def generate_trending_candidates(
  context: dict[str, Any],
  existing: set[str],
  variation_seed: int,
) -> list[dict[str, Any]]:
  from app.engine.seo_keyword_domain_engine import generate_domain_trending
  return generate_domain_trending(context, existing)


def _guess_cluster(kw: str, context: dict[str, Any]) -> str:
  k = kw.lower()
  for cluster in context.get("topic_clusters", []):
    if cluster.lower() in k or k.split()[0] in cluster.lower():
      return cluster
  return context.get("topic_clusters", ["General"])[0]


def validate_seo_quality(
  keywords: list[dict[str, Any]],
  context: dict[str, Any],
  seo_score: dict[str, Any],
) -> dict[str, Any]:
  issues: list[str] = []
  warnings: list[str] = []
  clusters_covered = {k.get("topic_cluster") for k in keywords}
  seed_clusters = set(context.get("topic_clusters", []))
  missing = seed_clusters - clusters_covered
  if missing:
    warnings.append(f"Topic clusters with few keywords: {', '.join(sorted(missing)[:5])}")
  cats = {k.get("category") for k in keywords}
  if "questions" not in cats:
    warnings.append("No question keywords generated — consider adding FAQ content.")
  if "local" not in cats:
    warnings.append("No local keywords — add target geography to seed.")
  if seo_score.get("overall", 0) < 50:
    issues.append("low_overall_seo_score")
  return {
    "valid": not issues,
    "issues": issues,
    "warnings": warnings,
    "checks": {
      "keyword_count": len(keywords),
      "topic_coverage": len(clusters_covered & seed_clusters) if seed_clusters else len(clusters_covered),
      "category_coverage": len(cats),
      "has_questions": "questions" in cats,
      "has_local": "local" in cats,
      "has_lsi": "lsi" in cats,
    },
    "score": seo_score.get("overall", 0),
  }


def build_recommendations(
  quality: dict[str, Any],
  context: dict[str, Any],
  opportunities: list[dict[str, Any]],
) -> list[str]:
  recs: list[str] = []
  if opportunities:
    top = opportunities[0].get("keyword", "")
    recs.append(f"Prioritize high-opportunity keyword: «{top}».")
  if context.get("topic_clusters"):
    recs.append(
      f"Create dedicated landing pages for top clusters: {', '.join(context['topic_clusters'][:4])}."
    )
  if not quality["checks"].get("has_local"):
    recs.append("Add target city/country to seed for stronger local SEO keywords.")
  if not quality["checks"].get("has_questions"):
    recs.append("Build FAQ schema content around generated question keywords.")
  recs.append("Metrics are AI estimates — validate with Google Search Console before budgeting.")
  recs.append("Use topic clusters to structure site architecture and internal linking.")
  return recs[:8]


def build_output_sections(
  *,
  context: dict[str, Any],
  seo_score: dict[str, Any],
  intent_dist: dict[str, int],
  entity_names: list[str],
  ranked: list[dict[str, Any]],
  keyword_categories: dict[str, list[dict[str, Any]]],
  topic_clusters: dict[str, list[dict[str, Any]]],
  opportunities: list[dict[str, Any]],
  geo: dict[str, Any],
  industry: dict[str, Any],
  language: dict[str, Any],
  brand: dict[str, Any],
  seed_intent: dict[str, Any],
  quality: dict[str, Any],
  recommendations: list[str],
  extra_competitor: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
  by_intent = {
    "commercial": [k for k in ranked if k.get("intent") == "commercial"],
    "transactional": [k for k in ranked if k.get("intent") == "transactional"],
    "informational": [k for k in ranked if k.get("intent") == "informational"],
    "navigational": [k for k in ranked if k.get("intent") == "navigational"],
  }
  competitor = [k for k in ranked if k.get("is_competitor") or "competitor" in " ".join(k.get("sources", []))]
  if not competitor and extra_competitor:
    competitor = extra_competitor[:10]
  trending = [k for k in ranked if k.get("is_trending") or k.get("trend") == "up"][:12]

  return {
    "context": {
      "seed": context.get("seed"),
      "brand": brand,
      "language": language,
      "geo": geo,
      "industry": industry,
      "primary_domain": context.get("primary_domain"),
      "domains": context.get("domains", []),
      "domain_category": context.get("domain_category"),
      "seed_intent": seed_intent,
      "topic_clusters": context.get("topic_clusters", []),
      "normalized": context.get("normalized"),
      "knowledge_graph": context.get("knowledge_graph"),
    },
    "seo_score": seo_score,
    "intent": {"distribution": intent_dist, "primary": seed_intent.get("primary_intent")},
    "entities": entity_names,
    "primary_keywords": keyword_categories.get("primary", []),
    "secondary_keywords": keyword_categories.get("secondary", []),
    "commercial_keywords": keyword_categories.get("commercial", []) + by_intent["commercial"][:5],
    "transactional_keywords": by_intent["transactional"],
    "informational_keywords": by_intent["informational"] + keyword_categories.get("questions", [])[:3],
    "long_tail_keywords": keyword_categories.get("long_tail", []),
    "question_keywords": keyword_categories.get("questions", []),
    "lsi_keywords": keyword_categories.get("lsi", []),
    "local_keywords": keyword_categories.get("local", []),
    "competitor_keywords": competitor,
    "trending_keywords": trending,
    "opportunity_keywords": opportunities,
    "keyword_clusters": topic_clusters,
    "metrics": {
      "volume_estimates": _metric_summary(ranked, "volume_label"),
      "difficulty_estimates": _metric_summary(ranked, "difficulty_label"),
      "cpc_estimates": _metric_summary(ranked, "cpc_label"),
      "competition_estimates": _metric_summary(ranked, "competition_label"),
      "metrics_source": METRICS_SOURCE,
    },
    "quality": quality,
    "recommendations": recommendations,
  }


def _metric_summary(items: list[dict[str, Any]], field: str) -> dict[str, int]:
  counts: dict[str, int] = {}
  for it in items:
    val = it.get(field, "unknown")
    counts[val] = counts.get(val, 0) + 1
  return counts

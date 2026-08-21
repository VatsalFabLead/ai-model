"""SEO Keyword Generator — pipeline stages, validation, and structured output."""

from __future__ import annotations

import re
from typing import Any

METRICS_SOURCE = "ai_estimate"

ARCHITECTURE_FLOW = [
  "input",
  "language_detection",
  "intent_classification",
  "industry_classification",
  "named_entity_recognition",
  "topic_extraction",
  "keyword_seed_generation",
  "semantic_expansion",
  "lsi_generation",
  "long_tail_generation",
  "question_generation",
  "search_intent_mapping",
  "competitor_topic_extraction",
  "trend_detection",
  "local_seo",
  "keyword_deduplication",
  "relevance_rerank",
  "volume_estimation",
  "difficulty_estimation",
  "cpc_estimation",
  "opportunity_scoring",
  "keyword_clustering",
  "ranking_prioritization",
  "quality_validation",
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
  brief = str(context.get("context_brief") or "").lower()
  hay = f"{low} {brief[:400]}"
  if any(w in hay for w in ("hire", "buy", "pricing", "cost", "agency", "company near me", "book now")):
    primary = "commercial"
  elif any(w in hay for w in ("how to", "what is", "why", "guide", "tutorial", "tips")):
    primary = "informational"
  elif any(w in hay for w in ("near me", "book", "order", "subscribe", "buy online")):
    primary = "transactional"
  elif context.get("is_brand_seed") and not context.get("topic_mode") and len((seed or "").split()) <= 4 and (
    context.get("domain_category") or ""
  ) in ("Technology", "Healthcare & Medical", "Business"):
    primary = "navigational"
  else:
    # Product / local / service briefs default to informational research intent
    category = context.get("domain_category") or ""
    if category in ("Technology", "Healthcare & Medical"):
      primary = "commercial"
    else:
      primary = "informational"
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


def detect_sub_intent(keyword: str, primary_intent: str) -> str:
  """Classify keywords into granular sub-intents."""
  low = (keyword or "").lower()
  if any(w in low for w in ("price", "cost", "pricing", "vs", "compare", "comparison", "quote", "fee", "ખર્ચ", "કિંમત", "कीमत", "precio", "tarif")):
    return "price_comparison"
  if any(w in low for w in ("fix", "error", "issue", "problem", "repair", "step by step", "how to", " guide", "સંભાળ", "देखभाल", "cómo reparar", "dépannage")):
    return "troubleshooting_guide"
  if any(w in low for w in ("features", "stack", "spec", "review", "demo", "overview", "સુવિધાઓ", "विशेषताएं", "características", "fonctionnalités")):
    return "feature_investigation"
  if any(w in low for w in ("hire", "company", "agency", "services", "developer", "provider", "સર્વિસ", "कंपनी", "servicios", "agence")):
    return "hiring_service"
  return "general_informational" if primary_intent == "informational" else f"{primary_intent}_general"


def detect_serp_feature_targets(keyword: str, intent: str, sub_intent: str) -> dict[str, Any]:
  """Identify target SERP overlay opportunities (Featured Snippet, PAA, Local Pack, Video)."""
  low = (keyword or "").lower()
  words = re.findall(r"\w+", low)

  is_question = any(w in low for w in ("how", "what", "why", "which", "when", "where", "who", "કેવી રીતે", "શું છે", "કેમ", "कैसे", "क्या है", "cómo", "qué", "comment")) or "?" in keyword
  is_definition = any(w in low for w in ("what is", "meaning", "definition", "overview", "શું છે", "क्या है", "qué es"))
  is_local = any(w in low for w in ("near me", "in ahmedabad", "in mumbai", "in delhi", "in surat", "in gujarat", "in india", "અમદાવાદમાં", "अहमदाबाद में"))
  is_video = any(w in low for w in ("tutorial", "video", "how to", "step by step", "demo", "ગાઈડ", "गाइड"))

  features: list[str] = []
  if is_question or is_definition:
    features.append("featured_snippet")
  if is_question:
    features.append("people_also_ask")
  if is_local:
    features.append("local_pack")
  if is_video:
    features.append("video_carousel")

  return {
    "featured_snippet_target": is_question or is_definition,
    "people_also_ask_target": is_question,
    "local_pack_target": is_local,
    "video_target": is_video,
    "serp_features": features or ["organic_result"],
  }


def detect_cannibalization_risks(keywords: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Calculate pairwise term overlap coefficient to alert cannibalization risks."""
  warnings: list[dict[str, Any]] = []
  for i in range(len(keywords)):
    k1 = keywords[i]["keyword"].lower()
    t1 = set(w for w in re.findall(r"\w+", k1) if len(w) >= 3)
    keywords[i]["cannibalization_risk"] = False
    keywords[i]["cannibalization_pair"] = None

    for j in range(i + 1, len(keywords)):
      k2 = keywords[j]["keyword"].lower()
      t2 = set(w for w in re.findall(r"\w+", k2) if len(w) >= 3)
      if not t1 or not t2:
        continue
      overlap = len(t1 & t2) / min(len(t1), len(t2))
      if overlap >= 0.70 and k1 != k2:
        keywords[i]["cannibalization_risk"] = True
        keywords[i]["cannibalization_pair"] = k2
        warnings.append({
          "keyword_a": keywords[i]["keyword"],
          "keyword_b": keywords[j]["keyword"],
          "overlap_score": round(overlap, 2),
          "recommendation": "Consolidate both keywords into a single master page to avoid SERP cannibalization.",
        })
        break
  return keywords, warnings


def generate_export_bundle(keywords: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
  """Generate pre-formatted CSV, Markdown Table, and Cluster Tree exports for SEO campaigns."""
  # CSV Export
  csv_lines = ["Keyword,Category,Primary Intent,Sub-Intent,Volume Estimate,Difficulty,CPC Estimate,SERP Targets,Cannibalization Risk"]
  for k in keywords:
    kw = f'"{k.get("keyword", "")}"'
    cat = k.get("category", "")
    intent = k.get("intent", "")
    sub_intent = k.get("sub_intent", "")
    vol = k.get("volume_label", k.get("volume_estimate", ""))
    diff = k.get("difficulty_label", k.get("difficulty_estimate", ""))
    cpc = k.get("cpc_label", k.get("cpc_estimate", ""))
    serp = ";".join(k.get("serp_features", []))
    can_risk = "YES" if k.get("cannibalization_risk") else "NO"
    csv_lines.append(f"{kw},{cat},{intent},{sub_intent},{vol},{diff},{cpc},{serp},{can_risk}")
  csv_export = "\n".join(csv_lines)

  # Markdown Table
  md_lines = [
    "| Keyword | Category | Sub-Intent | Volume | Difficulty | CPC | SERP Targets |",
    "|:---|:---|:---|:---|:---|:---|:---|",
  ]
  for k in keywords[:25]:
    kw = k.get("keyword", "")
    cat = k.get("category", "")
    sub_intent = k.get("sub_intent", "")
    vol = k.get("volume_label", k.get("volume_estimate", ""))
    diff = k.get("difficulty_label", k.get("difficulty_estimate", ""))
    cpc = k.get("cpc_label", k.get("cpc_estimate", ""))
    serp = ", ".join(k.get("serp_features", []))
    md_lines.append(f"| `{kw}` | {cat} | {sub_intent} | {vol} | {diff} | {cpc} | {serp} |")
  markdown_table = "\n".join(md_lines)

  # Cluster Tree Markdown
  clusters: dict[str, list[str]] = {}
  for k in keywords:
    c = k.get("topic_cluster") or "General"
    if c not in clusters:
      clusters[c] = []
    clusters[c].append(f"- **[{k.get('category', 'kw')}]** `{k.get('keyword')}` _({k.get('sub_intent', 'intent')})_")

  tree_lines = [f"# SEO Keyword Cluster Tree — {context.get('seed', 'Campaign')}\n"]
  for c_name, c_kws in clusters.items():
    tree_lines.append(f"### 📁 Topic Cluster: {c_name}")
    tree_lines.extend(c_kws)
    tree_lines.append("")
  cluster_tree_markdown = "\n".join(tree_lines)

  return {
    "csv_export": csv_export,
    "markdown_table": markdown_table,
    "cluster_tree_markdown": cluster_tree_markdown,
  }


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
  # Enrich ranked items with Sub-Intent, SERP Features, and Cannibalization
  for item in ranked:
    p_intent = item.get("intent", seed_intent.get("primary_intent", "informational"))
    s_intent = detect_sub_intent(item.get("keyword", ""), p_intent)
    item["sub_intent"] = s_intent
    serp_info = detect_serp_feature_targets(item.get("keyword", ""), p_intent, s_intent)
    item.update(serp_info)

  ranked, cannibalization_warnings = detect_cannibalization_risks(ranked)
  export_bundle = generate_export_bundle(ranked, context)

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
    "cannibalization_warnings": cannibalization_warnings,
    "export_bundle": export_bundle,
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

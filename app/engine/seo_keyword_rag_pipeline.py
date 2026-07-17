"""SEO Keyword Generator — production pipeline (open datasets, realistic keywords).

Metrics are AI estimates only — no fabricated exact search volume or CPC figures.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from typing import Any

from app.engine.keyword_discovery import discover_keywords
from app.engine.open_data_retrieval import OpenDoc
from app.engine.seo_content_domains import make_variation_seed

from app.engine.seo_keyword_enrichment import (
  ARCHITECTURE_FLOW,
  METRICS_SOURCE,
  OPEN_DATASET_TREE,
  build_output_sections,
  build_recommendations,
  classify_industry_domain,
  detect_country_region,
  detect_language,
  detect_seed_intent,
  expand_lsi_keywords,
  generate_competitor_keywords,
  generate_local_seo_keywords,
  generate_question_keywords,
  generate_trending_candidates,
  normalize_seed_keyword,
  normalize_seed_typos,
  recognize_brand_entity,
  validate_input,
  validate_seo_quality,
)
from app.engine.seo_keyword_domain_engine import (
  apply_domain_rules,
  expand_domain_keywords,
  knowledge_graph_lookup,
)
from app.engine.seo_keyword_domains import ALL_DOMAIN_NAMES, DOMAIN_CATALOG, DOMAIN_COUNT, classify_domains
from app.engine.seo_keyword_entity_kb import (
  detect_brand,
  disambiguate_entities,
  kb_primary_domain,
  run_named_entity_recognition,
)
from app.engine.seo_keyword_open_data import (
  DATASET_STACK,
  is_junk_open_keyword,
  retrieve_seo_keyword_data,
  terms_from_open_docs,
)
from app.engine.seo_keyword_relevance import (
  passes_topic_relevance,
  topic_relevance_score,
  topic_template_allowed,
)

GENERATOR_VERSION = "seo-keyword-rag-v5.3"

_INTENTS = ("informational", "commercial", "transactional", "navigational")
_CATEGORIES = (
  "primary", "secondary", "long_tail", "questions", "local", "commercial",
  "brand", "lsi", "transactional", "informational",
)

_COMMERCIAL_MARKERS = (
  "hire", "best", "top", "agency", "company", "services", "pricing", "cost",
  "developer", "development", "software", "consulting", "solutions",
)
_TRANSACTIONAL_MARKERS = ("near me", "online", "book", "order", "quote", "contact")
_INFORMATIONAL_MARKERS = ("how to", "what is", "why", "guide", "tutorial", "tips", "which", "when")
_QUESTION_STARTERS = ("how", "what", "why", "which", "when", "where", "who")

_LOCATIONS = (
  "india", "gujarat", "surat", "ahmedabad", "mumbai", "delhi", "bangalore",
  "usa", "uk", "canada", "australia", "dubai", "singapore",
)

_SERVICE_LIBRARY: dict[str, list[str]] = {
  "flutter": [
    "flutter app development",
    "flutter development company",
    "flutter development services",
    "hire flutter developers",
    "flutter mobile app development",
    "cross platform app development",
  ],
  "mobile": [
    "mobile app development company",
    "mobile app development services",
    "custom mobile app development",
    "ios android app development",
  ],
  "app": [
    "app development company",
    "custom app development",
    "food delivery app development",
    "healthcare app development",
  ],
  "erp": ["erp development company", "custom erp development", "erp software development"],
  "crm": ["crm development services", "custom crm development", "crm software company"],
  "ai": [
    "ai software development company",
    "ai development services",
    "custom ai solutions",
    "machine learning development company",
  ],
  "software": [
    "custom software development",
    "software development company",
    "enterprise software development",
    "software development services",
  ],
  "web": ["web development company", "web application development", "custom web development"],
  "hospital": ["hospital management software", "healthcare software development"],
  "food": ["food delivery app development", "restaurant app development"],
  "healthcare": [
    "healthcare app development",
    "hospital management software",
    "medical software development",
    "healthcare software development company",
  ],
  "telemedicine": ["telemedicine platform development", "telemedicine app development"],
  "hipaa": ["hipaa compliant software development", "hipaa compliant app development"],
  "machine": ["machine learning development company", "ai machine learning solutions"],
  "vision": ["medical image analysis software", "computer vision healthcare solutions"],
  "wearable": ["wearable health device app development", "iot healthcare solutions"],
  "firebase": ["firebase app development", "firebase mobile app development"],
  "python": ["python software development company", "python app development services"],
}

_DEFAULT_SERVICES = [
  "software development company",
  "mobile app development company",
  "custom software development",
  "app development services",
  "it solutions company",
  "digital transformation services",
]

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
_CPC_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "very_high": "Very High"}
_CPC_RANGES = {"low": "$0–1", "medium": "$1–3", "high": "$3–8", "very_high": "$8+"}
_LEVEL_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}
_TREND_ICONS = {"up": "📈", "stable": "➜", "down": "📉"}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun")

# Multi-word phrases matched longest-first against seed text
_KNOWN_ENTITY_PHRASES: list[tuple[str, str]] = [
  ("artificial intelligence", "Artificial Intelligence"),
  ("machine learning", "Machine Learning"),
  ("computer vision", "Computer Vision"),
  ("electronic health records", "Electronic Health Records"),
  ("remote patient monitoring", "Remote Patient Monitoring"),
  ("medical image analysis", "Medical Image Analysis"),
  ("mobile app development", "Mobile App Development"),
  ("cloud deployment", "Cloud Deployment"),
  ("wearable devices", "Wearable Devices"),
  ("iot healthcare", "IoT Healthcare"),
  ("telemedicine platform", "Telemedicine"),
  ("hipaa compliance", "HIPAA"),
  ("ai chatbot", "AI Chatbot"),
  ("healthcare startup", "Healthcare"),
  ("flutter", "Flutter"),
  ("firebase", "Firebase"),
  ("python", "Python"),
  ("healthcare", "Healthcare"),
  ("telemedicine", "Telemedicine"),
  ("hipaa", "HIPAA"),
  ("beauty product company", "Beauty"),
  ("beauty product", "Beauty"),
  ("beauty products", "Beauty"),
  ("cosmetics", "Cosmetics"),
  ("makeup", "Cosmetics"),
  ("skincare", "Beauty"),
  ("beauty", "Beauty"),
]

# Per-topic keyword templates — expanded when entity appears in seed
_TOPIC_TEMPLATES: dict[str, dict[str, list[str]]] = {
  "Beauty": {
    "primary": ["beauty products company", "cosmetics brand", "makeup brand india", "skincare products online"],
    "long_tail": ["vegan beauty products india", "cruelty free cosmetics brand", "affordable makeup brand online"],
    "questions": ["what is sugar cosmetics", "where to buy sugar makeup online", "is sugar cosmetics cruelty free"],
    "commercial": ["buy makeup online", "beauty products online shopping", "cosmetics ecommerce"],
    "lsi": ["lipstick collection", "makeup kit", "beauty essentials", "glow makeup"],
  },
  "Cosmetics": {
    "primary": ["cosmetics company", "makeup products", "lipstick brand", "foundation makeup"],
    "long_tail": ["best cosmetics brand in india", "professional makeup products"],
    "questions": ["what are the best cosmetics brands", "how to choose makeup products"],
    "commercial": ["cosmetics wholesale", "makeup distributor", "beauty products supplier"],
    "lsi": ["matte lipstick", "liquid foundation", "makeup palette"],
  },
  "Artificial Intelligence": {
    "primary": ["ai solutions", "artificial intelligence platform", "machine learning services", "ai software tools"],
    "long_tail": ["enterprise ai automation platform", "custom machine learning solutions", "ai powered business tools"],
    "questions": [
      "what is artificial intelligence",
      "how is ai used in business",
      "what are machine learning use cases",
    ],
    "commercial": ["ai consulting services", "ai development services", "hire ai developers"],
    "lsi": ["predictive analytics", "natural language processing", "generative ai"],
  },
  "Flutter": {
    "primary": ["flutter app development", "flutter mobile development", "cross platform flutter apps"],
    "long_tail": ["flutter ios android app development", "flutter development company india"],
    "questions": ["what is flutter app development", "how to build a flutter mobile app"],
    "commercial": ["hire flutter developers", "flutter development services", "flutter development company"],
    "lsi": ["dart programming", "cross platform mobile ui", "flutter widgets"],
  },
  "Python": {
    "primary": ["python development services", "python software development", "python web development"],
    "long_tail": ["python data science consulting", "python backend api development"],
    "questions": ["what is python used for", "how to learn python programming"],
    "commercial": ["hire python developers", "python development company"],
    "lsi": ["django flask", "python automation scripts", "data analysis python"],
  },
  "Computer Vision": {
    "primary": ["computer vision solutions", "image recognition software", "visual ai platform"],
    "long_tail": ["computer vision for quality inspection", "object detection software"],
    "questions": ["what is computer vision", "how does image recognition work"],
    "commercial": ["computer vision development services", "hire computer vision engineers"],
    "lsi": ["object detection", "image classification", "visual inspection ai"],
  },
  "Telemedicine": {
    "primary": ["telemedicine platform", "virtual healthcare platform", "telemedicine app development"],
    "long_tail": ["telemedicine platform development company", "remote healthcare consultation platform"],
    "questions": ["how to build a telemedicine platform", "what is a telemedicine platform"],
    "commercial": ["telemedicine software development", "telemedicine app development services"],
    "lsi": ["virtual care platform", "online doctor consultation app", "digital health platform"],
  },
  "Healthcare": {
    "primary": ["healthcare services", "medical clinic near me", "patient care services"],
    "long_tail": ["best healthcare providers nearby", "digital health services"],
    "questions": ["what is healthcare management", "how to choose a healthcare provider"],
    "commercial": ["healthcare providers near me", "book medical appointment"],
    "lsi": ["patient care", "clinical services", "health records"],
  },
  "HIPAA": {
    "primary": ["hipaa compliant software", "hipaa compliant app development", "hipaa compliance software"],
    "long_tail": ["hipaa compliant healthcare app development", "hipaa compliant telemedicine platform"],
    "questions": ["what is hipaa compliance", "how does hipaa compliance work", "how to make an app hipaa compliant"],
    "commercial": ["hipaa compliance consulting", "hipaa compliant software development company"],
    "lsi": ["health data privacy compliance", "phi security healthcare app", "hipaa security standards"],
  },
  "Machine Learning": {
    "primary": ["machine learning solutions", "ml model development", "predictive analytics platform"],
    "long_tail": ["custom machine learning model development", "ml consulting for business"],
    "questions": ["what is machine learning", "how is machine learning used in business"],
    "commercial": ["machine learning development company", "ml consulting services"],
    "lsi": ["supervised learning", "model training", "predictive modeling"],
  },
  "Electronic Health Records": {
    "primary": ["electronic health records software", "ehr software development", "ehr system development"],
    "long_tail": ["custom ehr software development", "cloud based ehr platform"],
    "questions": ["what are electronic health records", "how do ehr systems work"],
    "commercial": ["ehr software development company", "ehr implementation services"],
    "lsi": ["patient records management system", "digital medical records platform"],
  },
  "Remote Patient Monitoring": {
    "primary": ["remote patient monitoring software", "remote patient monitoring platform", "rpm healthcare software"],
    "long_tail": ["remote patient monitoring app development", "wearable remote patient monitoring"],
    "questions": ["what is remote patient monitoring", "how does remote patient monitoring work"],
    "commercial": ["remote patient monitoring solutions", "rpm software development services"],
    "lsi": ["continuous patient monitoring", "home health monitoring software"],
  },
  "AI Chatbot": {
    "primary": ["ai chatbot for healthcare", "medical ai chatbot", "healthcare chatbot development"],
    "long_tail": ["ai chatbot for patient support", "healthcare virtual assistant chatbot"],
    "questions": ["what is an ai chatbot in healthcare", "how do healthcare chatbots work"],
    "commercial": ["healthcare chatbot development services", "ai chatbot development company"],
    "lsi": ["patient engagement chatbot", "clinical chatbot assistant"],
  },
  "Medical Image Analysis": {
    "primary": ["medical image analysis software", "ai medical imaging", "radiology ai software"],
    "long_tail": ["ai powered medical image analysis", "automated medical image diagnosis"],
    "questions": ["what is medical image analysis", "how does ai analyze medical images"],
    "commercial": ["medical imaging software development", "ai radiology development services"],
    "lsi": ["medical scan ai analysis", "diagnostic imaging software"],
  },
  "Firebase": {
    "primary": ["firebase healthcare app", "firebase mobile app development", "firebase app development"],
    "long_tail": ["firebase telemedicine app development", "firebase backend healthcare app"],
    "questions": ["how to use firebase for healthcare apps"],
    "commercial": ["firebase development services", "hire firebase developers"],
    "lsi": ["firebase realtime database healthcare", "firebase authentication medical app"],
  },
  "Mobile App Development": {
    "primary": ["healthcare mobile app development", "medical mobile app development", "mobile health app development"],
    "long_tail": ["custom healthcare mobile app development company"],
    "questions": ["how to build a healthcare mobile app"],
    "commercial": ["mobile app development company", "healthcare app development services"],
    "lsi": ["mhealth app development", "patient mobile application"],
  },
  "Cloud Deployment": {
    "primary": ["cloud healthcare deployment", "healthcare cloud solutions", "cloud based healthcare software"],
    "long_tail": ["hipaa compliant cloud deployment healthcare", "aws healthcare cloud deployment"],
    "questions": ["how to deploy healthcare apps to the cloud"],
    "commercial": ["cloud deployment services healthcare", "healthcare cloud consulting"],
    "lsi": ["cloud infrastructure healthcare", "scalable healthcare cloud platform"],
  },
  "Wearable Devices": {
    "primary": ["wearable health devices app", "wearable healthcare technology", "health wearable app development"],
    "long_tail": ["wearable device integration healthcare app", "fitness wearable health monitoring"],
    "questions": ["how do wearable devices help healthcare"],
    "commercial": ["wearable app development services"],
    "lsi": ["smartwatch health monitoring", "wearable patient data collection"],
  },
  "IoT Healthcare": {
    "primary": ["iot healthcare solutions", "iot medical devices software", "healthcare iot platform"],
    "long_tail": ["iot remote patient monitoring system", "connected healthcare iot devices"],
    "questions": ["what is iot in healthcare", "how does iot improve healthcare"],
    "commercial": ["iot healthcare development company", "healthcare iot consulting"],
    "lsi": ["connected medical devices", "smart healthcare sensors"],
  },
}

_LOCAL_HEALTH_TEMPLATES = [
  "{topic} india",
  "{topic} startup india",
  "healthcare software company {loc}",
  "{topic} company {loc}",
  "hipaa compliant app development {loc}",
]


def effective_variation_seed(client_seed: int | None) -> int:
  base = make_variation_seed(client_seed)
  nonce = secrets.randbits(31) ^ (time.time_ns() & 0x7FFFFFFF)
  return make_variation_seed(base ^ nonce)


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _title_case_phrase(text: str) -> str:
  return " ".join(w.capitalize() for w in _clean(text).split())


def _kw_hash(keyword: str, seed: int) -> int:
  h = 0
  for ch in (keyword.lower() + str(seed)):
    h = (h * 31 + ord(ch)) & 0x7FFFFFFF
  return h


def _shuffle(items: list[Any], seed: int) -> list[Any]:
  out = list(items)
  for i in range(len(out) - 1, 0, -1):
    j = (seed + i * 7919) % (i + 1)
    out[i], out[j] = out[j], out[i]
  return out


def analyze_input(seed_keyword: str, *, language: str | None, tone: str) -> dict[str, Any]:
  seed = _clean(seed_keyword)
  tokens = [w for w in re.findall(r"\w+", seed.lower()) if len(w) > 2]
  return {
    "seed_keyword": seed,
    "token_count": len(tokens),
    "tokens": tokens,
    "language": language or "en",
    "tone": tone,
    "word_count": len(seed.split()),
    "is_long_tail": len(seed.split()) >= 3,
  }


def parse_input_context(
  seed_keyword: str,
  entities: list[dict[str, str]],
  docs: list[OpenDoc],
) -> dict[str, Any]:
  """Derive brand, topic clusters, and locations from seed."""
  seed = _clean(seed_keyword)
  tokens = [w for w in re.findall(r"\w+", seed) if len(w) > 1]
  low_tokens = [t.lower() for t in tokens]
  topic_mode = len(tokens) > 12 or "," in seed or " using " in seed.lower()

  topic_parts: list[str] = []
  for part in re.split(r"[,;]|\s+using\s+", seed, flags=re.I):
    part = _clean(part.lower())
    if part and len(part) > 3:
      topic_parts.append(part)

  topic_clusters: list[str] = []
  for ent in entities:
    cluster = ent.get("domain") or ent.get("cluster") or ent.get("name", "")
    if cluster and cluster not in topic_clusters:
      topic_clusters.append(cluster)
  if not topic_clusters and topic_parts:
    topic_clusters = [_title_case_phrase(topic_parts[0])]

  brand_parts = [t for t in tokens if t[0].isupper() and t.lower() not in {
    "artificial", "intelligence", "for", "using", "electronic", "health", "records",
    "beauty", "product", "products", "company", "cosmetics", "makeup",
  }]
  low_seed = seed.lower()
  if any(w in low_seed for w in ("beauty", "cosmetic", "makeup", "skincare")) and tokens:
    brand_name = _title_case_phrase(tokens[0])
  elif topic_mode and topic_parts:
    brand_name = _title_case_phrase(topic_parts[0][:80])
  elif brand_parts:
    brand_name = _title_case_phrase(" ".join(brand_parts[:2]))
  else:
    brand_name = _title_case_phrase(" ".join(tokens[:3])) if tokens else "Topic"

  locations: list[str] = []
  haystack = " ".join(low_tokens + topic_parts)
  for loc in _LOCATIONS:
    if loc in haystack:
      locations.append(loc.title())
  if not locations:
    locations = ["India"]

  is_brand_seed = not topic_mode and len(tokens) >= 3

  return {
    "brand_name": brand_name,
    "seed": seed,
    "topic_mode": topic_mode,
    "topic_parts": topic_parts[:24],
    "topic_clusters": topic_clusters,
    "entities": entities,
    "locations": list(dict.fromkeys(locations))[:6],
    "is_brand_seed": is_brand_seed,
    "core_topic": topic_clusters[0].lower() if topic_clusters else (topic_parts[0] if topic_parts else seed.lower()),
  }


def is_natural_keyword(keyword: str, context: dict[str, Any]) -> bool:
  """Reject awkward concatenations and open-data junk."""
  k = _clean(keyword.lower())
  if not k or len(k) < 3 or len(k) > 90:
    return False
  if is_junk_open_keyword(k, context):
    return False
  if not apply_domain_rules(k, context):
    return False
  words = k.split()
  industry = (context.get("industry") or {}).get("primary_industry", "")
  if industry in ("Beauty", "Cosmetics") and k == "sugar":
    return False
  if len(words) == 1 and industry not in ("Beauty", "Cosmetics", "Technology"):
    if k in ("sugar", "apple", "amazon", "target"):
      return False
  if len(words) < 2:
    return True

  seed = context.get("seed", "").lower()
  seed_tokens = {t for t in re.findall(r"\w+", seed) if len(t) > 2}
  # Long topic briefs: only treat first phrase tokens as brand anchor
  if context.get("topic_mode") and context.get("topic_parts"):
    seed_tokens = {t for t in re.findall(r"\w+", context["topic_parts"][0]) if len(t) > 2}
  kw_tokens = set(re.findall(r"\w+", k))

  # Reject when question prefix + full brand slug (3+ seed tokens all present)
  if words[0] in _QUESTION_STARTERS and len(seed_tokens) >= 3:
    if seed_tokens.issubset(kw_tokens) or sum(1 for t in seed_tokens if t in k) >= len(seed_tokens) - 1:
      return False

  awkward_prefixes = ("when to use", "how to", "best way to", "top")
  for prefix in awkward_prefixes:
    if k.startswith(prefix + " ") and len(seed_tokens) >= 3:
      tail = k[len(prefix):].strip()
      if sum(1 for t in seed_tokens if t in tail) >= 2:
        return False

  # Reject keywords that are just seed + single generic modifier with no service meaning
  generic_mods = {"technolab", "developers", "developer", "fablead"}
  if kw_tokens.issubset(seed_tokens | generic_mods) and len(words) >= 4:
    return False

  return True


def _hire_phrase(svc: str) -> str:
  low = svc.lower()
  if "flutter" in low:
    return "hire flutter developers"
  if "mobile" in low or "app" in low:
    return "hire mobile app developers"
  if "software" in low:
    return "hire software developers"
  if "erp" in low:
    return "hire erp developers"
  return "hire developers"


def generate_realistic_keywords(
  context: dict[str, Any],
  discovered: list[dict[str, Any]],
  docs: list[OpenDoc],
  *,
  count: int,
  variation_seed: int,
) -> list[dict[str, Any]]:
  """Domain-template keyword expansion with open-data terms."""
  open_terms = terms_from_open_docs(docs, context)
  rows = expand_domain_keywords(context, discovered, open_terms, count=count)
  # Legacy topic-cluster templates for tech/healthcare seeds still in catalog
  candidates: dict[str, dict[str, Any]] = {r["keyword"]: r for r in rows}
  clusters = context.get("topic_clusters") or [context.get("primary_domain", "General")]
  locations = context.get("locations") or ["India"]
  brand = context.get("brand_name", "")

  def add(
    kw: str,
    source: str,
    category: str,
    relevance: int,
    topic_cluster: str,
  ) -> None:
    k = _clean(kw.lower())
    if not is_natural_keyword(k, context):
      return
    row = candidates.get(k)
    if row is None:
      candidates[k] = {
        "keyword": k,
        "sources": [source],
        "category": category,
        "relevance": relevance,
        "topic_cluster": topic_cluster,
      }
    else:
      if source not in row["sources"]:
        row["sources"].append(source)
      row["relevance"] = max(row["relevance"], relevance)

  for cluster in clusters:
    if not topic_template_allowed(cluster, context):
      continue
    if cluster not in ALL_DOMAIN_NAMES:
      templates = _TOPIC_TEMPLATES.get(cluster, {})
      if templates:
        for kw in templates.get("primary", []):
          add(kw, f"topic:{cluster}", "primary", 92, cluster)
        for kw in templates.get("long_tail", []):
          add(kw, f"topic:{cluster}", "long_tail", 84, cluster)
    else:
      # Domain-named clusters still may have specialized packs
      templates = _TOPIC_TEMPLATES.get(cluster, {})
      if templates and topic_template_allowed(cluster, context):
        for kw in templates.get("primary", [])[:4]:
          add(kw, f"topic:{cluster}", "primary", 90, cluster)

  merged = list(candidates.values())
  # Re-score by seed/context relevance before truncation
  for row in merged:
    rel = topic_relevance_score(row["keyword"], context)
    row["relevance"] = int(0.55 * row.get("relevance", 50) + 0.45 * rel)
  merged = [r for r in merged if r["relevance"] >= 28 and passes_topic_relevance(r["keyword"], context)]
  merged.sort(key=lambda r: r["relevance"], reverse=True)
  return merged[: max(count * 3, 90)]


def _match_topic_cluster(keyword: str, clusters: list[str]) -> str:
  k = keyword.lower()
  for cluster in clusters:
    if cluster.lower() in k:
      return cluster
    templates = _TOPIC_TEMPLATES.get(cluster, {})
    for group in templates.values():
      if any(t in k or k in t for t in group):
        return cluster
  return clusters[0] if clusters else "General"


def generate_seed_variants(context: dict[str, Any], count: int, variation_seed: int) -> list[str]:
  variants: list[str] = []
  for cluster in context.get("topic_clusters", [])[:10]:
    if not topic_template_allowed(cluster, context):
      continue
    templates = _TOPIC_TEMPLATES.get(cluster, {})
    for kw in templates.get("primary", [])[:2]:
      variants.append(kw)
    variants.append(cluster.lower())
  if context.get("brand_name"):
    variants.append(context["brand_name"].lower())
  seed = _clean(str(context.get("normalized_seed") or context.get("seed_keyword") or ""))
  if isinstance(context.get("normalized"), dict) and not seed:
    seed = _clean(str(context["normalized"].get("normalized_seed") or ""))
  if seed:
    variants.append(seed)
  # Prefer seed-derived phrases from context brief
  brief = _clean(str(context.get("context_brief") or ""))
  if brief and brief.lower() != seed.lower():
    for part in re.split(r"[,;.]|\band\b", brief, flags=re.I):
      part = _clean(part.lower())
      part = re.sub(
        r"^(?:we\s+(?:are|run|sell|offer|provide)|our\s+\w+\s+is)\s+",
        "",
        part,
        flags=re.I,
      )
      words = [w for w in part.split() if len(w) > 2 and w not in {"the", "and", "for", "with", "from"}][:5]
      if 2 <= len(words) <= 5 and words[0] not in {"we", "our", "i"}:
        variants.append(" ".join(words))
  seen: set[str] = set()
  out: list[str] = []
  for v in _shuffle(variants, variation_seed):
    v = _clean(v.lower())
    if v and v not in seen and is_natural_keyword(v, context):
      seen.add(v)
      out.append(v)
  return out[: max(5, min(20, count))]


def extract_topics(
  seed_keyword: str,
  entities: list[dict[str, Any]],
  context: dict[str, Any],
) -> dict[str, Any]:
  """Topic extraction — clusters, entities, and seed phrases for expansion."""
  clusters = list(context.get("topic_clusters") or [])
  entity_topics = [e.get("name") or e.get("cluster") for e in entities if e.get("name") or e.get("cluster")]
  brand = (context.get("brand_name") or "").strip()
  primary_domain = context.get("primary_domain") or ""
  topics: list[str] = []
  seen: set[str] = set()
  for t in [seed_keyword, brand, primary_domain] + clusters + entity_topics:
    t = _clean(str(t or ""))
    if not t or len(t) < 2:
      continue
    low = t.lower()
    if low in seen:
      continue
    seen.add(low)
    topics.append(t)
  return {
    "topics": topics[:16],
    "primary_topic": topics[0] if topics else seed_keyword,
    "topic_clusters": clusters[:10],
    "entity_topics": entity_topics[:12],
  }


def semantic_expand_keywords(
  seed_keyword: str,
  docs: list[OpenDoc],
  existing: set[str],
  context: dict[str, Any],
  *,
  limit: int = 20,
) -> list[dict[str, Any]]:
  """Semantic expansion from Datamuse/open-doc related terms (embedding-like neighbors)."""
  terms = terms_from_open_docs(docs, context)
  # Prefer datamuse synonym docs
  for doc in docs:
    if doc.source == "datamuse" and doc.meta.get("terms"):
      terms = list(doc.meta["terms"]) + terms
  out: list[dict[str, Any]] = []
  seen = set(existing)
  seed_low = seed_keyword.lower()
  for term in terms:
    phrase = _clean(str(term).lower())
    if not phrase or len(phrase) < 3 or len(phrase) > 60:
      continue
    if phrase in seen or phrase == seed_low:
      continue
    if is_junk_open_keyword(phrase, context):
      continue
    if not passes_topic_relevance(phrase, context, min_score=24):
      continue
    # Prefer multi-word or related-to-seed phrases
    if " " not in phrase and phrase not in seed_low and seed_low.split()[0] not in phrase:
      relevance = 62
    else:
      relevance = 78
    relevance = int(0.5 * relevance + 0.5 * topic_relevance_score(phrase, context))
    seen.add(phrase)
    out.append({
      "keyword": phrase,
      "sources": ["semantic_expansion", "datamuse"],
      "category": "lsi",
      "relevance": relevance,
      "topic_cluster": context.get("primary_domain") or "General",
    })
    if len(out) >= limit:
      break
  return out


def generate_long_tail_keywords(
  context: dict[str, Any],
  existing: set[str],
  *,
  limit: int = 16,
) -> list[dict[str, Any]]:
  """Long-tail generation — multi-word intent phrases from seeds + templates."""
  seeds = [context.get("normalized_seed") or ""]
  if isinstance(context.get("normalized"), dict):
    seeds[0] = context["normalized"].get("normalized_seed") or seeds[0]
  if not seeds[0]:
    seeds[0] = str(context.get("seed_keyword") or "")
  seeds += list(context.get("topic_clusters") or [])[:4]
  brand = context.get("brand_name") or ""
  if brand:
    seeds.append(brand)
  locations = (context.get("locations") or ["india"])[:3]
  modifiers = (
    "for beginners", "best practices", "step by step", "near me",
    "cost", "examples", "guide 2026", "tips", "vs alternatives",
  )
  out: list[dict[str, Any]] = []
  seen = set(existing)
  for seed in seeds:
    base = _clean(str(seed).lower())
    if not base or len(base) < 2:
      continue
    candidates = [
      f"{base} {mod}" for mod in modifiers
    ] + [
      f"{base} in {loc.lower()}" for loc in locations
    ] + [
      f"how to {base}",
      f"what is {base}",
      f"best {base} tools",
    ]
    for phrase in candidates:
      phrase = _clean(phrase)
      if len(phrase.split()) < 3 or phrase in seen:
        continue
      if not is_natural_keyword(phrase, context):
        continue
      seen.add(phrase)
      out.append({
        "keyword": phrase,
        "sources": ["long_tail_generation"],
        "category": "long_tail",
        "relevance": 80,
        "topic_cluster": context.get("primary_domain") or "General",
      })
      if len(out) >= limit:
        return out
  return out


def _normalize_dedupe_key(keyword: str) -> str:
  """Near-duplicate key: lowercase, strip punctuation, collapse whitespace/plurals lightly."""
  k = (keyword or "").lower().strip()
  k = re.sub(r"[^\w\s]", " ", k)
  k = re.sub(r"\s+", " ", k).strip()
  # light plural trim for trailing s on last token
  parts = k.split()
  if parts and len(parts[-1]) > 3 and parts[-1].endswith("s") and not parts[-1].endswith("ss"):
    parts[-1] = parts[-1][:-1]
  return " ".join(parts)


def dedupe_keyword_candidates(
  candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Exact + near-duplicate keyword deduplication."""
  kept: list[dict[str, Any]] = []
  seen_exact: set[str] = set()
  seen_norm: set[str] = set()
  dropped_exact = 0
  dropped_near = 0
  for cand in candidates:
    kw = _clean(str(cand.get("keyword") or "").lower())
    if not kw:
      continue
    if kw in seen_exact:
      dropped_exact += 1
      continue
    norm = _normalize_dedupe_key(kw)
    if norm in seen_norm:
      dropped_near += 1
      continue
    seen_exact.add(kw)
    seen_norm.add(norm)
    cand = dict(cand)
    cand["keyword"] = kw
    kept.append(cand)
  return kept, {
    "input_count": len(candidates),
    "output_count": len(kept),
    "dropped_exact": dropped_exact,
    "dropped_near_duplicate": dropped_near,
  }


def extract_entities(seed: str, docs: list[OpenDoc]) -> list[dict[str, str]]:
  """Extract named topics/technologies from seed — comma lists and known phrases."""
  text = _clean(seed).lower()
  found: list[dict[str, str]] = []
  seen_clusters: set[str] = set()
  consumed_spans: list[tuple[int, int]] = []

  for phrase, cluster in sorted(_KNOWN_ENTITY_PHRASES, key=lambda x: -len(x[0])):
    start = 0
    while True:
      idx = text.find(phrase, start)
      if idx < 0:
        break
      end = idx + len(phrase)
      overlap = any(not (end <= s or idx >= e) for s, e in consumed_spans)
      if not overlap and cluster not in seen_clusters:
        seen_clusters.add(cluster)
        consumed_spans.append((idx, end))
        found.append({"name": cluster, "phrase": phrase, "cluster": cluster})
      start = idx + 1

  for part in re.split(r"[,;]|\s+using\s+", text, flags=re.I):
    part = _clean(part)
    if not part or len(part) < 4:
      continue
    matched = any(f["phrase"] in part or part in f["phrase"] for f in found)
    if not matched and 2 <= len(part.split()) <= 6:
      label = _title_case_phrase(part)
      if label not in seen_clusters:
        seen_clusters.add(label)
        found.append({"name": label, "phrase": part, "cluster": label})

  for d in docs[:6]:
    for w in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", d.title):
      if len(w) > 4 and w not in seen_clusters:
        seen_clusters.add(w)
        found.append({"name": w, "phrase": w.lower(), "cluster": w})

  return found[:24]


def extract_entity_names(entities: list[dict[str, str]]) -> list[str]:
  return [e["name"] for e in entities]


def classify_intent(keyword: str) -> str:
  k = keyword.lower()
  if any(h in k for h in _TRANSACTIONAL_MARKERS):
    return "transactional"
  if any(h in k for h in _COMMERCIAL_MARKERS):
    return "commercial"
  if any(k.startswith(h) for h in _INFORMATIONAL_MARKERS) or k.split()[0] in _QUESTION_STARTERS:
    return "informational"
  if any(h in k for h in ("official", "login", "website")):
    return "navigational"
  # Short product/topic phrases are informational by default (not commercial)
  words = k.split()
  if len(words) <= 3 and not any(m in k for m in ("services", "agency", "company", "hire", "pricing", "cost")):
    return "informational"
  return "commercial" if any(m in k for m in ("services", "agency", "company", "hire")) else "informational"


def classify_category(keyword: str, context: dict[str, Any], preset: str | None = None) -> str:
  if preset and preset in _CATEGORIES:
    return preset
  k = keyword.lower()
  brand = context.get("brand_name", "").lower()
  if brand and len(brand) > 3 and (k == brand or k.startswith(brand + " ")):
    return "brand"
  if k.split()[0] in _QUESTION_STARTERS or any(k.startswith(q) for q in _INFORMATIONAL_MARKERS):
    return "questions"
  if any(loc.lower() in k for loc in context.get("locations", [])) or "near me" in k:
    return "local"
  if any(h in k for h in _TRANSACTIONAL_MARKERS):
    return "transactional"
  if k.split()[0] in _QUESTION_STARTERS or any(k.startswith(h) for h in _INFORMATIONAL_MARKERS):
    return "informational"
  if any(m in k for m in ("hire", "best", "top", "services", "agency", "pricing", "cost", "consulting")):
    return "commercial"
  if len(k.split()) >= 6:
    return "long_tail"
  clusters = context.get("topic_clusters", [])
  if any(c.lower() in k for c in clusters[:3]):
    return "primary"
  return "secondary"


def estimate_volume_level(keyword: str, *, intent: str, category: str) -> str:
  words = keyword.split()
  if category == "brand":
    return "low"
  if category == "long_tail" or len(words) >= 6:
    return "very_low"
  if category == "questions":
    return "medium"
  if category == "primary" and len(words) <= 4:
    return "high"
  if intent == "commercial" and len(words) <= 4:
    return "medium"
  if len(words) <= 2:
    return "high"
  return "low" if len(words) >= 5 else "medium"


def estimate_difficulty_level(keyword: str, *, category: str, source_count: int) -> str:
  words = keyword.split()
  if category == "brand":
    return "low"
  if category == "long_tail":
    return "low"
  if len(words) <= 2:
    return "high"
  if len(words) >= 5:
    return "low"
  if source_count >= 2:
    return "medium"
  return "medium" if len(words) == 3 else "low"


def estimate_cpc_level(keyword: str, intent: str, category: str) -> str:
  k = keyword.lower()
  if intent == "transactional" or "hire" in k or "pricing" in k:
    return "high"
  if intent == "commercial" or category == "commercial":
    return "medium"
  if category == "brand":
    return "low"
  if category == "questions":
    return "low"
  return "medium"


def estimate_competition_level(difficulty: str, category: str) -> str:
  if category == "brand" or category == "long_tail":
    return "low"
  if difficulty == "high":
    return "high"
  if difficulty == "medium":
    return "medium"
  return "low"


def individual_trend(keyword: str, *, category: str, variation_seed: int) -> str:
  h = _kw_hash(keyword, variation_seed + 11)
  if category in {"primary", "commercial"}:
    return "up" if h % 3 != 2 else "stable"
  if category == "long_tail":
    return "stable"
  return ["up", "stable", "down"][h % 3]


def build_trend_monthly(trend: str, keyword: str, variation_seed: int) -> list[int]:
  """Estimated relative search interest by month (not absolute volume)."""
  h = _kw_hash(keyword, variation_seed + 17)
  base = 35 + (h % 25)
  values: list[int] = []
  for i in range(6):
    if trend == "up":
      values.append(min(100, base + i * (4 + h % 5)))
    elif trend == "down":
      values.append(max(10, base + (5 - i) * (3 + h % 4)))
    else:
      values.append(max(15, min(85, base + ((i % 3) - 1) * (2 + h % 3))))
  return values


def format_trend_chart(monthly: list[int]) -> str:
  parts = []
  for i, val in enumerate(monthly[:6]):
    bars = max(1, min(6, val // 15))
    parts.append(f"{_MONTHS[i]} {'█' * bars}")
  return " · ".join(parts)


def compute_opportunity_breakdown(item: dict[str, Any]) -> dict[str, Any]:
  vol = item["volume_estimate"]
  diff = item["difficulty_estimate"]
  cpc = item["cpc_estimate"]
  trend = item["trend"]
  intent = item["intent"]

  score = 50
  factors: list[str] = []
  vol_scores = {"very_low": 5, "low": 15, "medium": 25, "high": 35, "very_high": 40}
  diff_scores = {"low": 30, "medium": 18, "high": 5}
  score += vol_scores.get(vol, 15)
  score += diff_scores.get(diff, 10)
  if cpc in ("medium", "high", "very_high"):
    score += 10
    factors.append("Commercial CPC potential")
  if trend == "up":
    score += 12
    factors.append("Rising trend")
  elif trend == "stable":
    score += 4
  if intent in ("commercial", "transactional"):
    score += 8
    factors.append(f"{intent.title()} intent")

  if diff == "low" and vol in ("medium", "high"):
    factors.append("Good volume-to-difficulty balance")
  if vol == "very_low":
    factors.append("Limited search demand (estimated)")

  return {
    "volume": _VOLUME_LABELS.get(vol, vol),
    "difficulty": _LEVEL_LABELS.get(diff, diff),
    "cpc": _CPC_LABELS.get(cpc, cpc),
    "trend": trend,
    "intent": intent,
    "factors": factors[:4] or ["Balanced estimated profile"],
    "metrics_source": METRICS_SOURCE,
  }


def find_opportunities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  opps: list[dict[str, Any]] = []
  for it in items:
    breakdown = it.get("opportunity_breakdown") or compute_opportunity_breakdown(it)
    vol = it["volume_estimate"]
    diff = it["difficulty_estimate"]
    score = it.get("opportunity_score", 0)
    if diff in ("low", "medium") and vol in ("medium", "high", "low"):
      reason = breakdown["factors"][0] if breakdown.get("factors") else "balanced_estimate"
      opps.append({
        "keyword": it["keyword"],
        "opportunity_score": score,
        "reason": reason,
        "breakdown": breakdown,
      })
  opps.sort(key=lambda x: x["opportunity_score"], reverse=True)
  return opps[:12]


def categorize_keywords(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  groups: dict[str, list[dict[str, Any]]] = {cat: [] for cat in _CATEGORIES}
  for it in items:
    cat = it.get("category", "secondary")
    if cat not in groups:
      cat = "secondary"
    groups[cat].append(it)
  return {k: v for k, v in groups.items() if v}


def cluster_by_topic(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  groups: dict[str, list[dict[str, Any]]] = {}
  for it in items:
    cl = it.get("topic_cluster", "General")
    groups.setdefault(cl, []).append(it)
  return {k: v for k, v in sorted(groups.items()) if v}


def cluster_keywords(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Intent/category clusters plus topic cluster summary."""
  grouped = categorize_keywords(items)
  topic_grouped = cluster_by_topic(items)
  out = [
    {"cluster": name, "keywords": [it["keyword"] for it in kws[:12]], "count": len(kws)}
    for name, kws in grouped.items()
  ]
  for name, kws in topic_grouped.items():
    out.append({
      "cluster": f"topic:{name}",
      "topic": name,
      "keywords": [it["keyword"] for it in kws[:10]],
      "count": len(kws),
    })
  return out


def compute_seo_score(items: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
  if not items:
    return {"overall": 0, "coverage": 0, "diversity": 0, "category_coverage": {}, "topic_cluster_coverage": {}}
  categories = {it.get("category", "secondary") for it in items}
  topic_clusters = {it.get("topic_cluster", "General") for it in items}
  intents = {it["intent"] for it in items}
  trending = sum(1 for it in items if it["trend"] == "up")
  avg_opp = sum(it.get("opportunity_score", 0) for it in items) / len(items)
  seed_clusters = set(context.get("topic_clusters") or [])
  cluster_coverage = len(topic_clusters & seed_clusters) if seed_clusters else len(topic_clusters)
  diversity = len(categories) + len(topic_clusters)
  return {
    "overall": int(min(100, avg_opp * 0.5 + diversity * 4 + cluster_coverage * 3 + trending * 1.2)),
    "coverage": len(items),
    "diversity": diversity,
    "intent_diversity": len(intents),
    "topic_cluster_count": len(topic_clusters),
    "topic_cluster_coverage": {cl: sum(1 for it in items if it.get("topic_cluster") == cl) for cl in sorted(topic_clusters)},
    "trending_up": trending,
    "category_coverage": {cat: sum(1 for it in items if it.get("category") == cat) for cat in sorted(categories)},
    "metrics_source": METRICS_SOURCE,
  }


def assign_opportunity_scores(items: list[dict[str, Any]], variation_seed: int) -> None:
  """Spread scores 62–98 — avoid everything scoring 100."""
  ranked = sorted(
    items,
    key=lambda it: (it.get("relevance_score", 0), it.get("category") == "primary"),
    reverse=True,
  )
  n = len(ranked)
  for i, it in enumerate(ranked):
    pct = 1.0 - (i / max(n - 1, 1))
    base = int(62 + pct * 34)
    jitter = (_kw_hash(it["keyword"], variation_seed + i) % 7) - 3
    it["opportunity_score"] = min(98, max(62, base + jitter))


def select_diverse_keywords(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
  """Balanced pick across SEO categories AND topic clusters."""
  if len(items) <= count:
    return items
  by_cat: dict[str, list[dict[str, Any]]] = {cat: [] for cat in _CATEGORIES}
  by_topic: dict[str, list[dict[str, Any]]] = {}
  for it in items:
    cat = it.get("category", "secondary")
    by_cat.setdefault(cat, []).append(it)
    topic = it.get("topic_cluster", "General")
    by_topic.setdefault(topic, []).append(it)
  for bucket in list(by_cat.values()) + list(by_topic.values()):
    bucket.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)

  selected: list[dict[str, Any]] = []
  seen: set[str] = set()
  min_per_cat = max(1, count // (len(_CATEGORIES) + 2))
  cat_limits = {cat: min_per_cat for cat in _CATEGORIES}
  cat_limits["questions"] = max(4, min_per_cat)
  cat_limits["local"] = max(3, min_per_cat)
  cat_limits["commercial"] = max(3, min_per_cat)
  topics = list(by_topic.keys())

  for cat in _CATEGORIES:
    for it in by_cat.get(cat, [])[: cat_limits.get(cat, min_per_cat)]:
      if it["keyword"] not in seen:
        selected.append(it)
        seen.add(it["keyword"])

  per_topic = max(1, count // max(len(topics), 1))
  for topic in topics:
    for it in by_topic.get(topic, [])[:per_topic]:
      if it["keyword"] not in seen and len(selected) < count:
        selected.append(it)
        seen.add(it["keyword"])

  for it in rank_keywords([x for x in items if x["keyword"] not in seen]):
    if len(selected) >= count:
      break
    selected.append(it)
    seen.add(it["keyword"])
  return selected[:count]


def rank_keywords(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  vol_rank = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "very_low": 1}
  return sorted(
    items,
    key=lambda it: (
      it.get("topic_relevance", it.get("relevance_score", 0)),
      it.get("opportunity_score", 0),
      it.get("relevance_score", 0),
      vol_rank.get(it.get("volume_estimate", "low"), 1),
      -{"high": 3, "medium": 2, "low": 1}.get(it.get("difficulty_estimate", "medium"), 2),
    ),
    reverse=True,
  )


def build_keyword_row(
  keyword: str,
  *,
  context: dict[str, Any],
  sources: list[str],
  relevance: int,
  variation_seed: int,
  category: str | None = None,
  topic_cluster: str | None = None,
) -> dict[str, Any]:
  intent = classify_intent(keyword)
  cat = classify_category(keyword, context, preset=category)
  volume = estimate_volume_level(keyword, intent=intent, category=cat)
  difficulty = estimate_difficulty_level(keyword, category=cat, source_count=len(sources))
  cpc = estimate_cpc_level(keyword, intent, cat)
  competition = estimate_competition_level(difficulty, cat)
  trend = individual_trend(keyword, category=cat, variation_seed=variation_seed)
  monthly = build_trend_monthly(trend, keyword, variation_seed)
  cluster = topic_cluster or _match_topic_cluster(keyword, context.get("topic_clusters", []))
  topic_rel = topic_relevance_score(keyword, context)
  # Blend source relevance with seed/context relevance
  blended_relevance = int(0.5 * relevance + 0.5 * topic_rel)

  row: dict[str, Any] = {
    "keyword": keyword,
    "category": cat,
    "topic_cluster": cluster,
    "is_competitor": bool(sources and "competitor_generator" in sources),
    "is_trending": False,
    "volume_estimate": volume,
    "volume_label": _VOLUME_LABELS[volume],
    "volume_range": _VOLUME_RANGES[volume],
    "difficulty_estimate": difficulty,
    "difficulty_label": _LEVEL_LABELS[difficulty],
    "cpc_estimate": cpc,
    "cpc_label": _CPC_LABELS[cpc],
    "cpc_range": _CPC_RANGES[cpc],
    "competition_estimate": competition,
    "competition_label": _LEVEL_LABELS[competition],
    "trend": trend,
    "trend_icon": _TREND_ICONS.get(trend, "➜"),
    "trend_monthly": monthly,
    "trend_chart": format_trend_chart(monthly),
    "intent": intent,
    "relevance_score": blended_relevance,
    "topic_relevance": topic_rel,
    "sources": sources,
    "metrics_source": METRICS_SOURCE,
    "seo_score": int(min(100, blended_relevance * 0.35 + {"low": 25, "medium": 15, "high": 5}[difficulty] + (18 if trend == "up" else 6))),
    "opportunity_score": 0,
  }
  row["opportunity_breakdown"] = compute_opportunity_breakdown(row)
  return row


async def run_seo_keyword_pipeline(
  seed_keyword: str,
  *,
  context_text: str | None = None,
  variations: int = 10,
  tone: str = "informative",
  language: str | None = None,
  variation_seed: int | None = None,
  use_rag: bool = True,
  discover_web: bool = True,
  include_questions: bool = True,
  include_alphabet: bool = True,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  seed = effective_variation_seed(variation_seed)
  count = max(10, min(50, variations))
  seed_keyword = normalize_seed_typos(_clean(seed_keyword))
  context_brief = (context_text or "").strip() or seed_keyword
  # Analysis text = full context (brief) so NER/industry/topics use the business story
  analysis_text = context_brief if len(context_brief) >= len(seed_keyword) else seed_keyword
  stages: dict[str, Any] = {}

  stages["input"] = {
    "seed_keyword": seed_keyword,
    "context": context_brief[:500],
    "has_context": bool(context_text and context_text.strip() and context_text.strip() != seed_keyword),
    "requested_variations": count,
    "tone": tone,
  }

  input_val = validate_input(analysis_text if len(analysis_text) <= 2000 else seed_keyword)
  stages["input_validator"] = input_val
  if not input_val["valid"]:
    # Retry with short seed if long context fails validator length rules
    input_val = validate_input(seed_keyword)
    stages["input_validator"] = input_val
    if not input_val["valid"]:
      raise ValueError("; ".join(input_val["issues"]))

  corrected = normalize_seed_typos(seed_keyword)
  stages["spell_correction"] = {"original": seed_keyword, "corrected": corrected, "applied": corrected != seed_keyword}
  seed_keyword = corrected

  lang_info = detect_language(analysis_text, language)
  stages["language_detection"] = lang_info

  # NER on full context — richer entities from the business brief
  ner_entities = run_named_entity_recognition(analysis_text, [], known_phrases=_KNOWN_ENTITY_PHRASES)
  pre_context = parse_input_context(analysis_text, ner_entities, [])
  pre_context["kb_primary_domain"] = kb_primary_domain(analysis_text, ner_entities)
  pre_context["context_brief"] = context_brief
  pre_context["seed_keyword"] = seed_keyword

  domain_info = classify_domains(analysis_text, pre_context)
  pre_context.update({
    "primary_domain": domain_info["primary_domain"],
    "domains": domain_info["domains"],
    "domain_category": domain_info["category"],
  })
  industry = classify_industry_domain(analysis_text, pre_context)
  pre_context["industry"] = industry

  geo = detect_country_region(analysis_text, pre_context)
  stages["country_detection"] = geo
  pre_context["locations"] = geo.get("regions") or pre_context.get("locations", [])

  stages["named_entity_recognition"] = {"entities": ner_entities, "count": len(ner_entities)}

  docs: list[OpenDoc] = []
  rag_sources: list[str] = []
  open_data_meta: dict[str, Any] = {}
  if use_rag:
    try:
      docs, rag_sources, open_data_meta = await asyncio.wait_for(
        retrieve_seo_keyword_data(seed_keyword, pre_context, seed_int=seed, per_source=1, max_sources=10),
        timeout=12.0,
      )
    except asyncio.TimeoutError:
      docs = []

  entities = run_named_entity_recognition(analysis_text, docs, known_phrases=_KNOWN_ENTITY_PHRASES)
  disambiguated = disambiguate_entities(entities, analysis_text, pre_context)
  stages["entity_disambiguation"] = {"entities": disambiguated, "count": len(disambiguated)}

  context = parse_input_context(analysis_text, disambiguated, docs)
  context["seed_keyword"] = seed_keyword
  context["context_brief"] = context_brief
  context["kb_primary_domain"] = kb_primary_domain(analysis_text, disambiguated)
  domain_info = classify_domains(analysis_text, context)
  context.update({
    "primary_domain": domain_info["primary_domain"],
    "primary_domain_slug": domain_info.get("primary_domain_slug"),
    "primary_domain_id": domain_info.get("primary_domain_id"),
    "domains": domain_info["domains"],
    "domain_slugs": domain_info.get("domain_slugs", []),
    "domain_scores": domain_info.get("domain_scores", {}),
    "domain_category": domain_info["category"],
    "domain_flags": domain_info.get("flags", {}),
    "disambiguated_entities": disambiguated,
  })
  industry = classify_industry_domain(analysis_text, context)
  context["industry"] = industry
  context["locations"] = geo.get("regions") or context.get("locations", [])

  brand_info = detect_brand(analysis_text, disambiguated, context)
  context["brand_name"] = brand_info.get("brand_name", context.get("brand_name", ""))
  context["is_brand_seed"] = brand_info.get("is_brand_seed", context.get("is_brand_seed", False))
  stages["brand_detection"] = brand_info

  stages["industry_classification"] = industry
  entity_names = [e.get("name", "") for e in disambiguated if e.get("name")]

  seed_intent = detect_seed_intent(seed_keyword, context)
  stages["intent_classification"] = seed_intent

  topic_info = extract_topics(seed_keyword, disambiguated, context)
  context["extracted_topics"] = topic_info["topics"]
  # Prefer seed as primary topic when context brief is long
  if len(context_brief) > 80 and seed_keyword:
    topic_info = {**topic_info, "primary_topic": seed_keyword}
    if seed_keyword not in (topic_info.get("topics") or []):
      topic_info["topics"] = [seed_keyword] + list(topic_info.get("topics") or [])[:8]
  stages["topic_extraction"] = topic_info

  kg = knowledge_graph_lookup(seed_keyword, disambiguated, docs)
  stages["knowledge_graph_lookup"] = kg
  context["knowledge_graph"] = kg

  stages["domain_rule_engine"] = {
    "primary_domain": context["primary_domain"],
    "category": context.get("domain_category"),
    "blocked_cross_domain": True,
    "domain_count": DOMAIN_COUNT,
  }

  normalized = normalize_seed_keyword(seed_keyword, context)
  context["normalized"] = normalized
  context["normalized_seed"] = normalized.get("normalized_seed") or seed_keyword.lower()

  seed_variants = generate_seed_variants(context, count, seed)
  stages["keyword_seed_generation"] = {
    "seeds": seed_variants,
    "count": len(seed_variants),
    "primary_seed": context["normalized_seed"],
  }

  if use_rag:
    stages["open_data_retrieval"] = {
      **open_data_meta,
      "sources_used": rag_sources,
      "datasets": OPEN_DATASET_TREE,
      "dataset_stack": list(DATASET_STACK.keys()),
      "doc_count": len(docs),
    }
  else:
    stages["open_data_retrieval"] = {
      "skipped": True,
      "datasets": OPEN_DATASET_TREE,
      "dataset_stack": list(DATASET_STACK.keys()),
    }

  discovered: list[dict[str, Any]] = []
  discovery_meta: dict[str, Any] = {"enabled": discover_web, "sources_used": [], "queries_run": 0, "errors": []}
  if discover_web:
    try:
      discovery = await asyncio.wait_for(
        discover_keywords(
          seed_keyword,
          language=lang_info.get("language"),
          include_questions=include_questions,
          include_alphabet=include_alphabet,
          context=context,
        ),
        timeout=8.0,
      )
      discovered = discovery.get("keywords") or []
      discovery_meta.update({
        "sources_used": discovery.get("sources_used") or [],
        "queries_run": discovery.get("queries_run") or 0,
        "errors": discovery.get("errors") or [],
      })
    except asyncio.TimeoutError:
      discovery_meta["errors"] = ["discovery_timeout"]

  # Fold seed variants into discovery as high-relevance seeds
  for sv in seed_variants:
    discovered.append({"keyword": sv, "sources": ["keyword_seed_generation"], "relevance": 90})

  expanded = generate_realistic_keywords(context, discovered, docs, count=count, variation_seed=seed)
  existing_kws = {c["keyword"] for c in expanded}

  semantic_extra = semantic_expand_keywords(seed_keyword, docs, existing_kws, context, limit=max(12, count // 2))
  existing_kws |= {c["keyword"] for c in semantic_extra}
  stages["semantic_expansion"] = {
    "added": len(semantic_extra),
    "method": "datamuse_related_terms",
    "sources": sorted({s for c in semantic_extra for s in (c.get("sources") or [])}),
  }

  lsi_extra = expand_lsi_keywords(context, existing_kws)
  existing_kws |= {c["keyword"] for c in lsi_extra}
  long_tail_extra = generate_long_tail_keywords(context, existing_kws, limit=max(10, count))
  existing_kws |= {c["keyword"] for c in long_tail_extra}
  question_extra = generate_question_keywords(context, existing_kws) if include_questions else []
  existing_kws |= {c["keyword"] for c in question_extra}
  local_extra = generate_local_seo_keywords(context, existing_kws)
  existing_kws |= {c["keyword"] for c in local_extra}
  competitor_extra = generate_competitor_keywords(context, existing_kws)
  existing_kws |= {c["keyword"] for c in competitor_extra}
  trending_extra = generate_trending_candidates(context, existing_kws, seed)

  all_candidates = (
    expanded + semantic_extra + lsi_extra + long_tail_extra
    + question_extra + local_extra + competitor_extra + trending_extra
  )
  stages["keyword_expansion"] = {"candidates": len(expanded), "web_discovered": len(discovered)}
  stages["lsi_generation"] = {"added": len(lsi_extra)}
  stages["long_tail_generation"] = {"added": len(long_tail_extra)}
  stages["question_generation"] = {"added": len(question_extra)}
  stages["local_seo"] = {"added": len(local_extra), "markets": geo.get("countries", []), "optional": True}
  stages["competitor_topic_extraction"] = {"added": len(competitor_extra)}
  stages["trend_detection"] = {"added": len(trending_extra)}

  deduped_candidates, dedupe_meta = dedupe_keyword_candidates(all_candidates)
  # Final relevance gate — drop off-topic retrieval/expansion noise
  before_rel = len(deduped_candidates)
  deduped_candidates = [
    c for c in deduped_candidates
    if passes_topic_relevance(c.get("keyword", ""), context, min_score=26)
  ]
  dedupe_meta["dropped_off_topic"] = before_rel - len(deduped_candidates)
  stages["keyword_deduplication"] = dedupe_meta
  stages["relevance_rerank"] = {
    "method": "seed_context_token_overlap",
    "min_score": 26,
    "kept": len(deduped_candidates),
    "dropped_off_topic": dedupe_meta["dropped_off_topic"],
  }

  raw_items: list[dict[str, Any]] = []
  for i, cand in enumerate(deduped_candidates):
    kw = cand["keyword"]
    row = build_keyword_row(
      kw,
      context=context,
      sources=list(cand.get("sources") or ["pipeline_expansion"]),
      relevance=int(cand.get("relevance") or max(25, 85 - i)),
      variation_seed=seed + i * 17,
      category=cand.get("category"),
      topic_cluster=cand.get("topic_cluster"),
    )
    if cand.get("trend"):
      row["trend"] = cand["trend"]
      row["trend_icon"] = _TREND_ICONS.get(cand["trend"], "➜")
    if cand.get("is_trending"):
      row["is_trending"] = True
    if cand.get("is_competitor"):
      row["is_competitor"] = True
    raw_items.append(row)
    if len(raw_items) >= max(count * 4, 100):
      break

  assign_opportunity_scores(raw_items, seed)
  for it in raw_items:
    it["opportunity_breakdown"] = compute_opportunity_breakdown(it)

  intent_dist = {intent: sum(1 for it in raw_items if it["intent"] == intent) for intent in _INTENTS}
  stages["search_intent_mapping"] = {"distribution": intent_dist, "per_keyword": True}

  trend_agg = {"up": 0, "stable": 0, "down": 0}
  for it in raw_items:
    trend_agg[it["trend"]] = trend_agg.get(it["trend"], 0) + 1
  stages["trend_detection"] = {
    **stages.get("trend_detection", {}),
    **trend_agg,
    "metrics_source": METRICS_SOURCE,
  }

  vol_dist: dict[str, int] = {}
  diff_dist: dict[str, int] = {}
  cpc_dist: dict[str, int] = {}
  comp_dist: dict[str, int] = {}
  for it in raw_items:
    vol_dist[it["volume_label"]] = vol_dist.get(it["volume_label"], 0) + 1
    diff_dist[it["difficulty_label"]] = diff_dist.get(it["difficulty_label"], 0) + 1
    cpc_dist[it["cpc_label"]] = cpc_dist.get(it["cpc_label"], 0) + 1
    comp_dist[it["competition_label"]] = comp_dist.get(it["competition_label"], 0) + 1

  stages["volume_estimation"] = {"distribution": vol_dist, "metrics_source": METRICS_SOURCE}
  stages["difficulty_estimation"] = {"distribution": diff_dist, "metrics_source": METRICS_SOURCE}
  stages["cpc_estimation"] = {"distribution": cpc_dist, "metrics_source": METRICS_SOURCE}
  stages["competition_estimation"] = {"distribution": comp_dist, "metrics_source": METRICS_SOURCE}
  stages["opportunity_scoring"] = {"applied": True, "score_range": "62-98"}

  opportunities = find_opportunities(raw_items)
  ranked = select_diverse_keywords(rank_keywords(raw_items), count)
  keyword_categories = categorize_keywords(ranked)
  topic_clusters = cluster_by_topic(ranked)
  clusters = cluster_keywords(ranked)
  stages["keyword_clustering"] = {
    "clusters": len(clusters),
    "topic_clusters": list(topic_clusters.keys()),
    "category_counts": {k: len(v) for k, v in keyword_categories.items()},
  }
  stages["ranking_prioritization"] = {"ranked_count": len(ranked)}

  seo_score = compute_seo_score(ranked, context)
  stages["seo_score"] = seo_score

  quality = validate_seo_quality(ranked, context, seo_score)
  recommendations = build_recommendations(quality, context, opportunities)
  stages["quality_validation"] = quality

  ranked_keys = {r["keyword"] for r in ranked}
  competitor_rows: list[dict[str, Any]] = []
  for c in competitor_extra[:10]:
    if c["keyword"] not in ranked_keys:
      competitor_rows.append(build_keyword_row(
        c["keyword"], context=context, sources=c.get("sources", []),
        relevance=c.get("relevance", 70), variation_seed=seed + 999,
        category="commercial", topic_cluster=c.get("topic_cluster"),
      ))

  output = build_output_sections(
    context=context,
    seo_score=seo_score,
    intent_dist=intent_dist,
    entity_names=entity_names,
    ranked=ranked,
    keyword_categories=keyword_categories,
    topic_clusters=topic_clusters,
    opportunities=opportunities,
    geo=geo,
    industry=industry,
    language=lang_info,
    brand=brand_info,
    seed_intent=seed_intent,
    quality=quality,
    recommendations=recommendations,
    extra_competitor=competitor_rows,
  )
  stages["final_output"] = {"keyword_count": len(ranked), "sections": list(output.keys())}

  summary = {
    "high_volume_estimated": sum(1 for it in ranked if it["volume_estimate"] in ("high", "very_high")),
    "low_difficulty_estimated": sum(1 for it in ranked if it["difficulty_estimate"] == "low"),
    "trending_up": sum(1 for it in ranked if it["trend"] == "up"),
    "from_web": sum(1 for it in ranked if "web_discovery" in it.get("sources", [])),
    "opportunities": len(opportunities),
    "categories": len(keyword_categories),
    "topic_clusters": len(topic_clusters),
    "metrics_source": METRICS_SOURCE,
  }

  return {
    "generator_version": GENERATOR_VERSION,
    "seed_keyword": seed_keyword,
    "context": context_brief if context_brief != seed_keyword else None,
    "count": len(ranked),
    "variation_seed": seed,
    "keywords": ranked,
    "keyword_categories": keyword_categories,
    "topic_clusters": topic_clusters,
    "clusters": clusters,
    "opportunities": opportunities,
    "output": output,
    "summary": summary,
    "seo_score": seo_score,
    "recommendations": recommendations,
    "metrics_source": METRICS_SOURCE,
    "metrics_disclaimer": (
      "Search volume, CPC, difficulty, and competition are AI estimates — not data from "
      "Google Ads, Search Console, Ahrefs, or Semrush. Use qualitative labels for planning only."
    ),
    "discovery": discovery_meta,
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
      "domain_catalog_count": DOMAIN_COUNT,
    },
    "pipeline": {
      "input": analyze_input(seed_keyword, language=lang_info.get("language"), tone=tone),
      "context": context,
      "entities": entity_names,
      "entity_details": disambiguated,
      "topic_clusters": context["topic_clusters"],
      "primary_domain": context.get("primary_domain"),
      "primary_domain_slug": context.get("primary_domain_slug"),
      "domains": context.get("domains", []),
      "domain_category": context.get("domain_category"),
      "knowledge_graph": kg,
      "language": lang_info,
      "geo": geo,
      "industry": industry,
      "brand": brand_info,
      "seed_intent": seed_intent,
      "intent_distribution": intent_dist,
      "trend_aggregate": trend_agg,
      "retrieval": {"rag_sources": rag_sources, "document_count": len(docs), "enabled": use_rag},
    },
    "rag": {"enabled": use_rag, "sources_used": rag_sources},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "unlimited_outputs": True,
    "per_request_unique": True,
  }

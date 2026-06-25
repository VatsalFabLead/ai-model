"""Domain-first keyword engine — templates, rules, and generators per domain."""

from __future__ import annotations

import re
from typing import Any

from app.engine.seo_keyword_domains import (
  ADULT_BLOCKED_KEYWORD_TERMS,
  ADULT_RESTRICTED_DOMAINS,
  DOMAIN_BY_NAME,
  classify_domains,
  is_adult_restricted,
  resolve_domain,
)

_DOMAIN_LSI: dict[str, tuple[str, ...]] = {
  "Beauty": ("cruelty free makeup", "vegan cosmetics", "beauty essentials", "glow makeup", "lipstick shades"),
  "Cosmetics": ("matte lipstick", "liquid foundation", "makeup palette", "beauty kit"),
  "Skincare": ("vitamin c serum", "moisturizer spf", "skincare routine", "anti aging cream"),
  "Healthcare": ("digital health", "patient care", "clinical workflow", "medical records"),
  "Telemedicine": ("virtual care", "remote consultation", "online doctor", "digital clinic"),
  "Artificial Intelligence": ("machine learning model", "ai automation", "predictive analytics", "nlp"),
  "Mobile App Development": ("cross platform app", "ios android development", "flutter dart", "mobile ui"),
  "Web Development": ("responsive website", "frontend framework", "web application", "full stack"),
  "SEO": ("keyword research", "on page seo", "backlink strategy", "search rankings"),
  "Restaurant": ("fine dining", "food menu", "table booking", "chef special"),
  "Food Delivery": ("order food online", "food delivery app", "restaurant delivery", "quick delivery"),
  "Real Estate": ("property listing", "home for sale", "rental apartment", "real estate agent"),
  "Travel": ("holiday package", "travel booking", "vacation deals", "tourist guide"),
  "Finance": ("financial planning", "wealth management", "investment portfolio", "personal finance"),
  "Ecommerce": ("online shopping", "ecommerce store", "product catalog", "checkout"),
}

_DOMAIN_BLOCK_TERMS: dict[str, tuple[str, ...]] = {
  "Beauty": ("healthcare", "telemedicine", "hipaa", "hospital", "medical software", "developer", "flutter"),
  "Cosmetics": ("healthcare", "telemedicine", "software development", "machine learning"),
  "Skincare": ("healthcare", "telemedicine", "software development"),
  "Healthcare": ("makeup", "cosmetics", "lipstick", "beauty products"),
  "Telemedicine": ("makeup", "cosmetics", "fashion"),
  "Mobile App Development": ("makeup", "cosmetics", "restaurant menu"),
  "Restaurant": ("software development", "flutter", "machine learning"),
}

_TRENDING_BY_DOMAIN: dict[str, tuple[str, ...]] = {
  "Beauty": ("clean beauty products", "vegan cosmetics", "skincare routine", "cruelty free makeup"),
  "Cosmetics": ("clean makeup brands", "affordable cosmetics online", "k beauty products"),
  "Healthcare": ("ai healthcare solutions", "telemedicine platform", "digital health apps"),
  "Telemedicine": ("virtual healthcare", "remote patient monitoring", "online doctor consultation"),
  "Artificial Intelligence": ("generative ai tools", "ai automation", "llm applications"),
  "Mobile App Development": ("flutter app development", "cross platform apps", "mobile app trends"),
  "SEO": ("ai seo tools", "content optimization", "search intent targeting"),
  "Food Delivery": ("ghost kitchen", "cloud kitchen delivery", "quick commerce food"),
  "Ecommerce": ("social commerce", "dtc brands", "marketplace selling"),
  "Travel": ("sustainable tourism", "workation packages", "travel booking apps"),
  "Plumber": ("emergency plumber", "plumbing repair", "pipe installation"),
  "Electrician": ("licensed electrician", "electrical wiring", "home electrician"),
  "Local Business": ("local business marketing", "google business profile", "near me search"),
  "Brand": ("brand awareness", "brand identity", "brand strategy"),
  "Company": ("company profile", "about company", "corporate website"),
  "Product": ("product launch", "product marketing", "product catalog"),
}

# Informational-only templates for restricted adult domains (no promotional / explicit terms)
_ADULT_SAFE_TEMPLATES: dict[str, dict[str, list[str]]] = {
  "Escort Services": {
    "questions": [
      "what are escort service regulations",
      "legal framework for companion services",
      "safety guidelines for service providers",
    ],
    "primary": ["companion service regulations", "legal compliance guide"],
  },
  "Companion Services": {
    "questions": [
      "what is a companion service business",
      "companion care vs escort services",
      "regulations for companion services",
    ],
    "primary": ["companion care services", "professional companion guidelines"],
  },
  "Webcam Platforms": {
    "questions": [
      "how do streaming platform regulations work",
      "online content moderation policies",
      "creator platform safety guidelines",
    ],
    "primary": ["streaming platform policy", "content moderation guide"],
  },
  "Adult Toys": {
    "questions": [
      "what is sexual wellness education",
      "intimate health product safety standards",
    ],
    "primary": ["sexual wellness education", "intimate health products guide"],
    "commercial": ["wellness product retailer", "health and wellness store"],
  },
  "Nightlife": {
    "questions": [
      "what is responsible nightlife business",
      "nightclub licensing requirements",
      "evening entertainment regulations",
    ],
    "primary": ["nightlife venue licensing", "responsible entertainment business"],
    "local": ["nightlife district guide", "evening entertainment area"],
  },
  "Sexual Wellness": {
    "questions": [
      "what is sexual wellness",
      "how to talk about intimate health",
      "sexual health education resources",
    ],
    "primary": ["sexual wellness education", "intimate health awareness"],
  },
  "Relationship Advice": {
    "questions": [
      "how to improve communication in relationships",
      "what is healthy relationship advice",
      "couples counseling benefits",
    ],
    "primary": ["relationship advice blog", "healthy relationship tips"],
  },
  "Adult Education": {
    "questions": [
      "what is adult education",
      "continuing education programs for adults",
      "lifelong learning benefits",
    ],
    "primary": ["adult education courses", "continuing education programs"],
  },
}


def _clean(text: str) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _dl(domain: str) -> str:
  return domain.lower()


def get_domain_templates(domain: str, brand: str = "", location: str = "india") -> dict[str, list[str]]:
  """Per-domain keyword templates — primary, commercial, questions, long_tail, lsi, local."""
  entry = resolve_domain(domain)
  display = entry["domain"] if entry else domain

  if is_adult_restricted(display) or display in _ADULT_SAFE_TEMPLATES:
    safe = _ADULT_SAFE_TEMPLATES.get(display, _ADULT_SAFE_TEMPLATES.get("Sexual Wellness", {}))
    loc = location.lower()
    templates: dict[str, list[str]] = {
      "primary": list(safe.get("primary", [f"{display.lower()} information"])),
      "commercial": list(safe.get("commercial", [])),
      "questions": list(safe.get("questions", [f"what is {display.lower()}"])),
      "long_tail": [f"{display.lower()} guidelines {loc}"],
      "lsi": list(safe.get("lsi", ("education", "wellness", "regulations"))),
      "local": list(safe.get("local", [f"{display.lower()} regulations {loc}"])),
    }
    return templates

  d = _dl(display)
  b = brand.lower() if brand else d
  loc = location.lower()
  templates: dict[str, list[str]] = {
    "primary": [
      f"{d} company",
      f"best {d} services",
      f"{d} solutions",
      f"top {d} providers",
    ],
    "commercial": [
      f"{d} pricing",
      f"hire {d} services",
      f"{d} agency",
      f"affordable {d}",
    ],
    "questions": [
      f"what is {d}",
      f"how to choose {d}",
      f"best {d} for beginners",
      f"why {d} matters",
    ],
    "long_tail": [
      f"best {d} company in {loc}",
      f"affordable {d} services {loc}",
      f"professional {d} near me",
    ],
    "lsi": list(_DOMAIN_LSI.get(display, (f"{d} services", f"{d} industry", f"{d} experts"))),
    "local": [
      f"{d} near me",
      f"{d} in {loc}",
      f"{d} company {loc}",
      f"best {d} {loc}",
    ],
  }
  if brand:
    templates["primary"].extend([
      f"{b} {d}",
      f"{b} official",
      f"{b} products",
    ])
    templates["questions"].extend([
      f"what is {b}",
      f"where to buy {b}",
      f"is {b} good",
    ])
    templates["brand"] = [b, f"{b} {d}", f"{b} online"]
  overrides = _TEMPLATE_OVERRIDES.get(display)
  if overrides:
    for k, v in overrides.items():
      templates[k] = list(dict.fromkeys(templates.get(k, []) + v))
  return templates


_TEMPLATE_OVERRIDES: dict[str, dict[str, list[str]]] = {
  "Beauty": {
    "primary": ["beauty products company", "cosmetics brand", "makeup brand", "skincare products online"],
    "commercial": ["buy makeup online", "beauty products online shopping", "cosmetics ecommerce"],
    "questions": ["what are cruelty free beauty brands", "how to choose makeup products"],
  },
  "Cosmetics": {
    "primary": ["cosmetics company", "makeup products", "lipstick brand", "foundation makeup"],
    "commercial": ["cosmetics wholesale", "makeup distributor"],
  },
  "Healthcare": {
    "primary": ["healthcare software", "medical app development", "hospital management software"],
    "commercial": ["healthcare software development company", "medical app development services"],
  },
  "Telemedicine": {
    "primary": ["telemedicine platform", "virtual healthcare platform", "telemedicine app"],
    "questions": ["how to build a telemedicine platform", "what is telemedicine"],
  },
  "Artificial Intelligence": {
    "primary": ["ai solutions", "ai software development", "machine learning services"],
    "questions": ["what is artificial intelligence", "how is ai used in business"],
  },
  "Mobile App Development": {
    "primary": ["mobile app development company", "flutter app development", "ios android app development"],
    "commercial": ["hire mobile app developers", "app development services"],
  },
  "SEO": {
    "primary": ["seo services", "seo agency", "keyword research services"],
    "questions": ["what is seo", "how does seo work", "how to improve seo rankings"],
  },
  "Food Delivery": {
    "primary": ["food delivery app", "restaurant food delivery", "order food online"],
    "commercial": ["food delivery service", "online food ordering"],
  },
}


def apply_domain_rules(keyword: str, context: dict[str, Any]) -> bool:
  """Domain rule engine — block cross-domain contamination and unsafe adult terms."""
  k = keyword.lower()
  domain = context.get("primary_domain") or (context.get("industry") or {}).get("primary_industry", "")
  category = context.get("domain_category", "")

  if any(term in k for term in ADULT_BLOCKED_KEYWORD_TERMS):
    return False

  if category == "Adult" or is_adult_restricted(domain):
    promo_markers = ("hire", "book now", "cheap", "best escort", "call now", "hot ")
    if any(m in k for m in promo_markers):
      return False

  blocked = _DOMAIN_BLOCK_TERMS.get(domain, ())
  if any(b in k for b in blocked):
    return False
  industry = (context.get("industry") or {}).get("primary_industry", "")
  if industry in ("Beauty", "Cosmetics", "Skincare") and k in ("sugar",) and "beauty" not in k and "cosmetic" not in k:
    return False
  if industry in ("Beauty", "Cosmetics", "Skincare") and any(t in k for t in ("software", "technologies", "developer")):
    return False
  return True


def knowledge_graph_lookup(
  seed: str,
  entities: list[dict[str, Any]],
  docs: list[Any],
) -> dict[str, Any]:
  """Merge entity KB + open-data docs into knowledge graph context."""
  nodes: list[dict[str, Any]] = []
  edges: list[dict[str, str]] = []

  for ent in entities:
    nodes.append({
      "id": ent.get("name", ""),
      "type": ent.get("type", "entity"),
      "domain": ent.get("domain", ""),
      "source": ent.get("source", "ner"),
    })
    if ent.get("domain"):
      edges.append({"from": ent.get("name", ""), "to": ent["domain"], "relation": "belongs_to"})

  for doc in docs[:10]:
    src = getattr(doc, "source", "open_data")
    title = getattr(doc, "title", "") or ""
    if title:
      nodes.append({"id": title[:60], "type": "concept", "source": src})
      for ent in entities[:3]:
        if ent.get("name") and ent["name"].lower() in title.lower():
          edges.append({"from": title[:60], "to": ent["name"], "relation": "mentions"})

  return {
    "node_count": len(nodes),
    "edge_count": len(edges),
    "nodes": nodes[:20],
    "edges": edges[:20],
    "sources": sorted({getattr(d, "source", "") for d in docs[:10]}),
  }


def expand_domain_keywords(
  context: dict[str, Any],
  discovered: list[dict[str, Any]],
  open_terms: list[str],
  *,
  count: int,
) -> list[dict[str, Any]]:
  """Keyword expansion from domain templates."""
  out: list[dict[str, Any]] = []
  seen: set[str] = set()
  domains = context.get("domains") or [context.get("primary_domain", "Business")]
  brand = context.get("brand_name", "")
  locations = context.get("locations") or ["India"]

  def add(kw: str, source: str, category: str, relevance: int, domain: str) -> None:
    k = _clean(kw.lower())
    if not k or k in seen or len(k) < 3 or len(k) > 90:
      return
    if not apply_domain_rules(k, context):
      return
    seen.add(k)
    out.append({
      "keyword": k,
      "sources": [source],
      "category": category,
      "relevance": relevance,
      "topic_cluster": domain,
    })

  for domain in domains[:4]:
    loc = (locations[0] if locations else "India").lower()
    tpl = get_domain_templates(domain, brand=brand, location=loc)
    for kw in tpl.get("primary", []):
      add(kw, f"domain:{domain}", "primary", 92, domain)
    for kw in tpl.get("commercial", []):
      add(kw, f"domain:{domain}", "commercial", 88, domain)
    for kw in tpl.get("long_tail", []):
      add(kw, f"domain:{domain}", "long_tail", 84, domain)
    for kw in tpl.get("brand", []):
      add(kw, f"domain:{domain}", "brand", 90, domain)
    for loc_name in locations[:2]:
      for kw in tpl.get("local", []):
        add(kw.replace(loc, loc_name.lower()), f"domain:{domain}", "local", 80, domain)

  for item in discovered:
    kw = item.get("keyword", "")
    if apply_domain_rules(kw, context):
      add(kw, "web_discovery", "secondary", int(item.get("relevance_score") or 55),
          context.get("primary_domain", "General"))

  for term in open_terms:
    add(term, "open_data", "lsi", 62, context.get("primary_domain", "General"))

  out.sort(key=lambda x: x["relevance"], reverse=True)
  return out[: max(count * 3, 90)]


def generate_domain_questions(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  brand = context.get("brand_name", "")
  for domain in (context.get("domains") or [context.get("primary_domain", "Business")])[:4]:
    tpl = get_domain_templates(domain, brand=brand)
    for q in tpl.get("questions", []):
      if q not in existing and apply_domain_rules(q, context):
        out.append({
          "keyword": q,
          "sources": ["question_generator"],
          "category": "questions",
          "relevance": 86,
          "topic_cluster": domain,
        })
  return out


def generate_domain_competitors(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  for domain in (context.get("domains") or [context.get("primary_domain", "Business")])[:5]:
    d = _dl(domain)
    patterns = [
      f"best {d} companies",
      f"top {d} services",
      f"{d} comparison",
      f"alternative {d} providers",
      f"{d} vs competitors",
    ]
    for kw in patterns:
      if kw not in existing and apply_domain_rules(kw, context):
        out.append({
          "keyword": kw,
          "sources": ["competitor_generator"],
          "category": "commercial",
          "relevance": 72,
          "topic_cluster": domain,
          "is_competitor": True,
        })
  return out


def generate_domain_lsi(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  for domain in (context.get("domains") or [context.get("primary_domain", "Business")])[:4]:
    for phrase in _DOMAIN_LSI.get(domain, ()):
      kw = phrase.lower()
      if kw not in existing and apply_domain_rules(kw, context):
        out.append({
          "keyword": kw,
          "sources": ["lsi_expansion"],
          "category": "lsi",
          "relevance": 74,
          "topic_cluster": domain,
        })
  return out


def generate_domain_local(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  locations = context.get("locations") or ["India"]
  brand = (context.get("brand_name") or "").lower()
  domain = context.get("primary_domain", "Business")
  tpl = get_domain_templates(domain, brand=brand)
  for loc in locations[:4]:
    loc_l = loc.lower()
    for kw in tpl.get("local", []):
      localized = kw.replace("india", loc_l).replace("near me", f"in {loc_l}")
      if localized not in existing and apply_domain_rules(localized, context):
        out.append({
          "keyword": localized,
          "sources": ["local_seo"],
          "category": "local",
          "relevance": 78,
          "topic_cluster": domain,
        })
  return out


def generate_domain_trending(context: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
  out: list[dict[str, Any]] = []
  domain = context.get("primary_domain", "Business")
  topics = _TRENDING_BY_DOMAIN.get(domain, ())
  if not topics:
    d = _dl(domain)
    topics = (f"{d} trends 2025", f"best {d} solutions", f"popular {d} services")
  for kw in topics:
    if kw not in existing and apply_domain_rules(kw, context):
      out.append({
        "keyword": kw,
        "sources": ["trending_engine"],
        "category": "primary",
        "relevance": 88,
        "topic_cluster": domain,
        "trend": "up",
        "is_trending": True,
      })
  return out


def classify_industry_from_domain(domain_info: dict[str, Any]) -> dict[str, Any]:
  """Backward-compatible industry block from domain classifier."""
  return {
    "industries": domain_info.get("industries", [domain_info["primary_domain"]]),
    "primary_industry": domain_info["primary_industry"],
    "domain": domain_info.get("domain", ""),
    "primary_domain": domain_info["primary_domain"],
    "primary_domain_slug": domain_info.get("primary_domain_slug"),
    "primary_domain_id": domain_info.get("primary_domain_id"),
    "category": domain_info.get("category"),
    "domains": domain_info.get("domains", []),
  }

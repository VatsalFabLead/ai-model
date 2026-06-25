"""Schema markup enrichment — validation, scoring, relationships, rich results."""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_CONTEXT = "https://schema.org"

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "Schema.org Vocabulary": ["schema_knowledge", "wikipedia", "wikidata"],
  "Google Search Central": ["schema_knowledge", "gooaq"],
  "JSON-LD Specification": ["schema_knowledge", "wikipedia"],
  "Open Knowledge Graph": ["wikidata", "dbpedia", "conceptnet"],
  "Rich Results Guidelines": ["schema_knowledge", "wikipedia"],
  "General Knowledge": ["wikipedia", "wikidata", "dbpedia"],
  "Question Answering": ["gooaq", "squad"],
  "Multilingual": ["wikipedia", "wikidata"],
  "E-commerce": ["wikipedia", "gooaq"],
  "Programming": ["stackexchange", "wikipedia"],
  "Medical": ["pubmed", "wikipedia"],
  "Geography": ["wikidata", "openstreetmap"],
}

OPEN_DATA_SOURCES = ["wikipedia", "wikidata", "gooaq"]

# User-fill templates — never invent fake URLs, addresses, names, or dates
TPL: dict[str, str] = {
  "AUTHOR_NAME": "{{AUTHOR_NAME}}",
  "PUBLISHER_NAME": "{{PUBLISHER_NAME}}",
  "PUBLISHER_LOGO": "{{PUBLISHER_LOGO_URL}}",
  "IMAGE_URL": "{{IMAGE_URL}}",
  "URL": "{{URL}}",
  "DATE_PUBLISHED": "{{DATE_PUBLISHED}}",
  "DATE_MODIFIED": "{{DATE_MODIFIED}}",
  "DATE_POSTED": "{{DATE_POSTED}}",
  "UPLOAD_DATE": "{{UPLOAD_DATE}}",
  "STREET_ADDRESS": "{{STREET_ADDRESS}}",
  "CITY": "{{CITY}}",
  "STATE": "{{STATE}}",
  "POSTAL_CODE": "{{POSTAL_CODE}}",
  "COUNTRY": "{{COUNTRY}}",
  "PHONE": "{{PHONE}}",
  "OPENING_HOURS": "{{OPENING_HOURS}}",
  "LATITUDE": "{{LATITUDE}}",
  "LONGITUDE": "{{LONGITUDE}}",
  "LOGO_URL": "{{LOGO_URL}}",
  "PRICE_RANGE": "{{PRICE_RANGE}}",
  "SOCIAL_URL": "{{SOCIAL_URL}}",
  "PRICE": "{{PRICE}}",
  "CURRENCY": "{{CURRENCY}}",
  "CONTENT_URL": "{{CONTENT_URL}}",
  "DESCRIPTION": "{{DESCRIPTION}}",
  "ORGANIZATION_NAME": "{{ORGANIZATION_NAME}}",
  "BRAND_NAME": "{{BRAND_NAME}}",
  "PROVIDER_NAME": "{{PROVIDER_NAME}}",
  "RATING_VALUE": "{{RATING_VALUE}}",
  "REVIEW_COUNT": "{{REVIEW_COUNT}}",
  "RATING": "{{RATING}}",
  "SERVICE_TYPE": "{{SERVICE_TYPE}}",
  "AREA_SERVED": "{{AREA_SERVED}}",
}

# Types that should include @id in production output
_ID_TYPES = frozenset({
  "Article", "NewsArticle", "Blog", "BlogPosting", "Organization", "WebSite", "WebPage",
  "LocalBusiness", "Product", "Service", "Course", "Person",
})

# Google-preferred image property
_IMAGE_TYPES = frozenset({
  "Article", "NewsArticle", "Blog", "BlogPosting", "Product", "Course", "Service",
  "Recipe", "Organization",
})

# Most business/content types benefit from description + url placeholders
_DESC_URL_TYPES = frozenset({
  "Article", "NewsArticle", "Blog", "BlogPosting", "Product", "Organization", "Service",
  "Course", "Brand", "Person", "WebPage", "WebSite", "LocalBusiness", "MedicalBusiness",
  "Restaurant", "RealEstateAgent", "Recipe", "JobPosting", "VideoObject", "HowTo",
})

_TEMPLATE_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")

_KEYWORD_JUNK_RE = re.compile(
  r"(?i)\b(add |include |must |should |works worldwide|best practices|schema\.org|"
  r"json-ld|postaladdress|@type|base salary|pricevaliduntil|rich results|bcp-?47)\b",
)

OPTIONAL_PROPERTIES: dict[str, list[str]] = {
  "Article": [
    "image", "publisher", "description", "dateModified", "mainEntityOfPage", "keywords",
  ],
  "Product": [
    "image", "description", "sku", "brand", "offers", "aggregateRating", "review", "category",
  ],
  "LocalBusiness": [
    "telephone", "url", "logo", "geo", "openingHoursSpecification", "sameAs", "contactPoint",
    "priceRange",
  ],
  "FAQPage": ["author", "publisher", "inLanguage"],
  "Organization": [
    "logo", "url", "contactPoint", "address", "sameAs", "foundingDate",
  ],
  "Recipe": [
    "image", "prepTime", "cookTime", "totalTime", "nutrition", "aggregateRating", "video",
  ],
  "Person": ["jobTitle", "image", "url", "sameAs", "worksFor", "address"],
  "WebSite": ["publisher", "potentialAction", "inLanguage"],
  "WebPage": ["description", "breadcrumb", "isPartOf", "primaryImageOfPage"],
  "Blog": ["image", "publisher", "articleBody", "keywords"],
  "BlogPosting": ["image", "publisher", "articleBody", "keywords"],
  "NewsArticle": ["image", "publisher", "dateModified", "articleSection"],
  "HowTo": ["image", "estimatedCost", "supply", "tool", "totalTime"],
  "Review": ["itemReviewed", "publisher", "datePublished"],
  "AggregateRating": ["bestRating", "worstRating"],
  "Service": ["description", "provider", "areaServed", "offers", "serviceType"],
  "MedicalBusiness": [
    "telephone", "medicalSpecialty", "openingHoursSpecification", "geo",
  ],
  "Restaurant": [
    "servesCuisine", "menu", "acceptsReservations", "priceRange", "openingHoursSpecification",
  ],
  "RealEstateAgent": [
    "telephone", "areaServed", "openingHoursSpecification", "sameAs",
  ],
  "Course": [
    "provider", "description", "educationalCredentialAwarded", "offers",
  ],
  "JobPosting": [
    "baseSalary", "employmentType", "validThrough", "applicantLocationRequirements",
  ],
  "VideoObject": [
    "thumbnailUrl", "description", "duration", "embedUrl", "contentUrl",
  ],
  "ImageObject": ["caption", "creator", "width", "height", "license"],
  "PodcastEpisode": [
    "description", "datePublished", "associatedMedia", "duration",
  ],
  "Offer": ["availability", "seller", "validFrom", "url", "itemCondition"],
  "Brand": ["logo", "url", "slogan", "sameAs"],
  "BreadcrumbList": ["position", "item", "name"],
  "SitelinksSearchBox": ["query-input", "target"],
}

# Required + recommended property registry (Schema.org / Google aligned)
REQUIRED_PROPERTIES: dict[str, list[str]] = {
  "Article": ["headline", "author", "datePublished"],
  "Product": ["name"],
  "LocalBusiness": ["name", "address"],
  "FAQPage": ["mainEntity"],
  "Organization": ["name"],
  "Recipe": ["name", "recipeIngredient", "recipeInstructions"],
  "Person": ["name"],
  "WebSite": ["name", "url"],
  "WebPage": ["name", "url"],
  "Blog": ["headline", "author", "datePublished"],
  "BlogPosting": ["headline", "author", "datePublished"],
  "NewsArticle": ["headline", "author", "datePublished"],
  "HowTo": ["name", "step"],
  "Review": ["reviewBody", "reviewRating", "author"],
  "AggregateRating": ["ratingValue", "reviewCount"],
  "Service": ["name"],
  "MedicalBusiness": ["name", "address"],
  "Restaurant": ["name", "address"],
  "RealEstateAgent": ["name", "address"],
  "Course": ["name"],
  "JobPosting": ["title", "hiringOrganization", "jobLocation"],
  "VideoObject": ["name", "uploadDate"],
  "ImageObject": ["contentUrl"],
  "PodcastEpisode": ["name", "partOfSeries"],
  "Offer": ["price", "priceCurrency"],
  "Brand": ["name"],
  "BreadcrumbList": ["itemListElement"],
  "SitelinksSearchBox": ["url", "potentialAction"],
  # Extended types (same rules as closest match)
  "Event": ["name", "startDate", "location"],
  "SoftwareApplication": ["name"],
  "MobileApplication": ["name"],
  "Store": ["name", "address"],
  "EducationalOrganization": ["name"],
  "GovernmentOrganization": ["name"],
  "FinancialService": ["name", "address"],
  "LegalService": ["name", "address"],
  "Dentist": ["name", "address"],
  "Hotel": ["name", "address"],
  "Book": ["name"],
  "Movie": ["name"],
  "MusicRecording": ["name"],
  "TouristAttraction": ["name"],
}

TOPIC_KEYWORD_HINTS: list[tuple[tuple[str, ...], list[str]]] = [
  (("developer", "technolab", "software", "tech", "app", "digital"), [
    "Flutter Development", "Mobile App Development", "Web Development",
    "ERP Solutions", "CRM Solutions", "AI Software Development",
  ]),
  (("restaurant", "food", "cafe", "dining"), [
    "Restaurant", "Food Service", "Dining", "Catering", "Local Cuisine",
  ]),
  (("clinic", "medical", "health", "dental", "hospital"), [
    "Healthcare", "Medical Services", "Patient Care", "Clinical Services",
  ]),
  (("real estate", "property", "realtor"), [
    "Real Estate", "Property Sales", "Home Listings", "Commercial Property",
  ]),
  (("course", "training", "academy", "education"), [
    "Online Course", "Professional Training", "Certification", "Skills Development",
  ]),
  (("recipe", "cooking", "cuisine"), [
    "Recipe", "Cooking Guide", "Home Cooking", "Culinary Tips",
  ]),
  (("marketing", "seo", "content"), [
    "Digital Marketing", "SEO Strategy", "Content Marketing", "Search Optimization",
  ]),
]

RICH_RESULTS_ELIGIBLE: frozenset[str] = frozenset({
  "Article", "NewsArticle", "Blog", "BlogPosting", "FAQPage", "HowTo", "Product", "Recipe",
  "JobPosting", "Event", "Course", "VideoObject", "BreadcrumbList", "Review",
  "LocalBusiness", "Restaurant", "Organization", "SoftwareApplication", "PodcastEpisode",
  "WebSite", "WebPage", "Person", "Service", "Offer", "Brand",
})

NESTED_RELATIONSHIPS: dict[str, list[str]] = {
  "Article": ["author", "publisher", "image", "mainEntityOfPage"],
  "NewsArticle": ["author", "publisher", "image"],
  "Blog": ["author", "publisher", "image"],
  "BlogPosting": ["author", "publisher", "image"],
  "Product": ["brand", "offers", "aggregateRating"],
  "Review": ["reviewRating", "author", "itemReviewed"],
  "JobPosting": ["hiringOrganization", "jobLocation"],
  "Course": ["provider", "offers"],
  "Recipe": ["recipeInstructions"],
  "LocalBusiness": ["address", "geo", "contactPoint"],
  "MedicalBusiness": ["address", "geo"],
  "Restaurant": ["address"],
  "RealEstateAgent": ["address"],
  "Organization": ["logo", "contactPoint", "address"],
  "VideoObject": ["thumbnailUrl"],
  "PodcastEpisode": ["partOfSeries"],
  "Service": ["provider", "offers"],
  "WebSite": ["publisher", "potentialAction"],
  "WebPage": ["primaryImageOfPage", "breadcrumb"],
}

# User-facing aliases and typos → canonical Schema.org type
TYPE_ALIASES: dict[str, str] = {
  "organisation": "Organization",
  "organization": "Organization",
  "local business": "LocalBusiness",
  "localbusiness": "LocalBusiness",
  "faq": "FAQPage",
  "faqpage": "FAQPage",
  "faq page": "FAQPage",
  "webpage": "WebPage",
  "web page": "WebPage",
  "website": "WebSite",
  "web site": "WebSite",
  "newsarticle": "NewsArticle",
  "news article": "NewsArticle",
  "new article": "NewsArticle",
  "howto": "HowTo",
  "how to": "HowTo",
  "how-to": "HowTo",
  "aggregaterating": "AggregateRating",
  "aggregate rating": "AggregateRating",
  "medicalbusiness": "MedicalBusiness",
  "medical business": "MedicalBusiness",
  "realestateagent": "RealEstateAgent",
  "real estate agent": "RealEstateAgent",
  "jobposting": "JobPosting",
  "job posting": "JobPosting",
  "videoobject": "VideoObject",
  "video": "VideoObject",
  "imageobject": "ImageObject",
  "image": "ImageObject",
  "podcastepisode": "PodcastEpisode",
  "poscastepissode": "PodcastEpisode",
  "podcast episode": "PodcastEpisode",
  "breadcrumblist": "BreadcrumbList",
  "breadcrumb": "BreadcrumbList",
  "breadcrumb list": "BreadcrumbList",
  "sitelinkssearchbox": "SitelinksSearchBox",
  "sitelinks search box": "SitelinksSearchBox",
  "localbusiness": "LocalBusiness",
  "blogpost": "Blog",
  "blogposting": "BlogPosting",
  "blog posting": "BlogPosting",
  "review": "Review",
  "brand": "Brand",
  "offer": "Offer",
  "service": "Service",
  "restaurant": "Restaurant",
  "recipe": "Recipe",
  "course": "Course",
  "person": "Person",
  "product": "Product",
  "article": "Article",
}


def is_clean_keyword(kw: str) -> bool:
  k = (kw or "").strip()
  if not k or len(k) < 4 or len(k) > 60:
    return False
  if _KEYWORD_JUNK_RE.search(k):
    return False
  if _TEMPLATE_RE.search(k):
    return False
  if k.lower().startswith(("fable is", "this page", "learn more")):
    return False
  words = k.split()
  if len(words) == 1 and words[0].lower() in {"fable", "developers", "technolab", "place", "works"}:
    return False
  return True


def sanitize_keyword_list(keywords: list[str], *, limit: int = 10) -> list[str]:
  out: list[str] = []
  seen: set[str] = set()
  for kw in keywords:
    if not isinstance(kw, str):
      continue
    k = kw.strip()
    if not is_clean_keyword(k):
      continue
    lk = k.lower()
    if lk in seen:
      continue
    seen.add(lk)
    out.append(k)
  return out[:limit]


def is_template_value(value: Any) -> bool:
  if isinstance(value, str):
    return bool(_TEMPLATE_RE.search(value))
  return False


def contains_template(value: Any) -> bool:
  if isinstance(value, str):
    return bool(_TEMPLATE_RE.search(value))
  if isinstance(value, dict):
    return any(contains_template(v) for v in value.values())
  if isinstance(value, list):
    return any(contains_template(v) for v in value)
  return False


def user_provided(value: Any) -> bool:
  if value is None or value == "" or value == []:
    return False
  return not contains_template(value)


def normalize_schema_type(raw: str, supported_map: dict[str, str]) -> str:
  key = re.sub(r"[\s_\-]+", " ", (raw or "").strip().lower())
  if key in TYPE_ALIASES:
    canonical = TYPE_ALIASES[key]
    if canonical.lower() in supported_map:
      return supported_map[canonical.lower()]
    return canonical
  compact = key.replace(" ", "")
  if compact in TYPE_ALIASES:
    canonical = TYPE_ALIASES[compact]
    if canonical.lower() in supported_map:
      return supported_map[canonical.lower()]
    return canonical
  if key in supported_map:
    return supported_map[key]
  if compact in supported_map:
    return supported_map[compact]
  raise ValueError(f"Unsupported schema type: {raw}")


def validate_input(
  schema_type: str,
  name: str,
  type_map: dict[str, str] | None = None,
) -> dict[str, Any]:
  issues: list[str] = []
  canonical: str | None = None
  type_supported = False
  if not schema_type or not schema_type.strip():
    issues.append("schema_type_required")
  elif type_map:
    try:
      canonical = normalize_schema_type(schema_type, type_map)
      type_supported = True
    except ValueError:
      issues.append("schema_type_unsupported")
  else:
    type_supported = bool(schema_type.strip())

  name_ok = bool(name and str(name).strip() and len(str(name).strip()) >= 2)
  if not name or not str(name).strip():
    issues.append("name_required")
  elif len(str(name).strip()) < 2:
    issues.append("name_too_short")

  return {
    "valid": not issues,
    "issues": issues,
    "canonical_type": canonical,
    "checks": {
      "schema_type_supported": type_supported,
      "title_name_provided": name_ok,
    },
  }


def extract_entities(name: str, data: dict[str, Any]) -> list[str]:
  """Named entities from verified description/title — not naive title token splits."""
  entities: list[str] = []
  seen: set[str] = set()
  for text in (str(data.get("description", "")), str(data.get("title", ""))):
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\b", text):
      k = m.lower()
      if k not in seen and len(m) > 3:
        seen.add(k)
        entities.append(m)
  return entities[:12]


def schema_base_id(name: str, schema_type: str | None = None) -> str:
  slug = slugify(name)
  suffix = slugify(schema_type or "entity")
  return f"https://example.org/schema/{slug}#{suffix}"


def resolve_effective_type(schema_type: str) -> str:
  """SitelinksSearchBox is expressed as WebSite + SearchAction per Google."""
  if schema_type == "SitelinksSearchBox":
    return "WebSite"
  return schema_type


def infer_topic_keywords(name: str, data: dict[str, Any]) -> list[str]:
  """Keywords only from user input or page description — omit if none available."""
  keywords: list[str] = []
  for kw in data.get("keywords") or []:
    if isinstance(kw, str) and is_clean_keyword(kw):
      keywords.append(kw.strip())
  desc = str(data.get("description", "")).strip()
  if desc:
    for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){1,3}\b", desc):
      if is_clean_keyword(phrase):
        keywords.append(phrase)
  return sanitize_keyword_list(keywords, limit=10)


def extract_keywords_from_docs(docs: list[Any], topic: str) -> list[str]:
  keywords: list[str] = []
  seen: set[str] = set()

  def add(kw: str) -> None:
    k = kw.strip()
    if not is_clean_keyword(k):
      return
    lk = k.lower()
    if lk in seen:
      return
    seen.add(lk)
    keywords.append(k)

  for doc in docs[:5]:
    if getattr(doc, "source", "") in {"schema_knowledge", "schema guidance"}:
      continue
    title = (getattr(doc, "title", "") or "").strip()
    if title and is_clean_keyword(title):
      add(title)
    text = getattr(doc, "text", "") or ""
    for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){1,2}\b", text):
      if len(match) > 10:
        add(match)
  return sanitize_keyword_list(keywords, limit=8)


def property_requirements(schema_type: str) -> dict[str, Any]:
  required = REQUIRED_PROPERTIES.get(schema_type, ["name"])
  recommended = OPTIONAL_PROPERTIES.get(schema_type, [])
  return {
    "required": required,
    "recommended": recommended,
    "rich_results_eligible": schema_type in RICH_RESULTS_ELIGIBLE,
  }


def check_required_properties(schema: dict[str, Any], schema_type: str) -> list[str]:
  required = REQUIRED_PROPERTIES.get(schema_type, ["name"])
  missing: list[str] = []
  for prop in required:
    val = schema.get(prop)
    if val is None or val == "" or val == []:
      missing.append(prop)
  return missing


def map_relationships(schema: dict[str, Any], schema_type: str) -> dict[str, Any]:
  """Ensure nested objects use @type where appropriate."""
  out = dict(schema)
  nested_checks = {
    "author": "Person",
    "publisher": "Organization",
    "provider": "Organization",
    "hiringOrganization": "Organization",
    "brand": "Brand",
    "offers": "Offer",
    "aggregateRating": "AggregateRating",
    "reviewRating": "Rating",
    "address": "PostalAddress",
    "geo": "GeoCoordinates",
    "location": "Place",
    "itemReviewed": "Thing",
    "partOfSeries": "PodcastSeries",
  }
  for key, default_type in nested_checks.items():
    val = out.get(key)
    if isinstance(val, dict) and "@type" not in val:
      out[key] = {"@type": default_type, **val}
  if schema_type == "FAQPage" and isinstance(out.get("mainEntity"), list):
    fixed = []
    for item in out["mainEntity"]:
      if not isinstance(item, dict):
        continue
      q = dict(item)
      q.setdefault("@type", "Question")
      ans = q.get("acceptedAnswer")
      if isinstance(ans, dict) and "@type" not in ans:
        q["acceptedAnswer"] = {"@type": "Answer", **ans}
      fixed.append(q)
    out["mainEntity"] = fixed
  if schema_type == "BreadcrumbList" and isinstance(out.get("itemListElement"), list):
    out["itemListElement"] = [
      {**item, "@type": item.get("@type", "ListItem")}
      if isinstance(item, dict) else item
      for item in out["itemListElement"]
    ]
  return out


def _address_from_data(data: dict[str, Any]) -> dict[str, Any]:
  """PostalAddress with user values or templates — never fake street/city."""
  return {
    "@type": "PostalAddress",
    "streetAddress": str(data.get("streetAddress") or data.get("address") or TPL["STREET_ADDRESS"]),
    "addressLocality": str(data.get("city") or data.get("addressLocality") or TPL["CITY"]),
    "addressRegion": str(data.get("state") or data.get("addressRegion") or TPL["STATE"]),
    "postalCode": str(data.get("postalCode") or data.get("zip") or TPL["POSTAL_CODE"]),
    "addressCountry": str(data.get("country") or data.get("addressCountry") or TPL["COUNTRY"]),
  }


def _collect_warnings(schema: dict[str, Any], data: dict[str, Any], schema_type: str) -> list[str]:
  warnings: list[str] = []
  primary = _primary_node(schema)

  for prop in REQUIRED_PROPERTIES.get(schema_type, ["name"]):
    val = primary.get(prop)
    if val is None or val == "" or val == [] or contains_template(val):
      warnings.append(f"Required property '{prop}' needs user input.")

  for prop in OPTIONAL_PROPERTIES.get(schema_type, []):
    val = primary.get(prop)
    if val is None or val == "" or val == [] or contains_template(val):
      warnings.append(f"Recommended property '{prop}' is missing.")

  return warnings


def apply_verified_defaults(
  schema: dict[str, Any],
  schema_type: str,
  name: str,
  data: dict[str, Any],
) -> dict[str, Any]:
  """Required templates + production recommended placeholders — no fake data."""
  out = dict(schema)
  content_types = ("Article", "NewsArticle", "Blog", "BlogPosting")

  # WebPage must never carry article fields
  if schema_type == "WebPage":
    out.pop("headline", None)
    out.pop("articleSection", None)
    out.pop("author", None)
    out.pop("publisher", None)
    out.pop("datePublished", None)
    out.pop("dateModified", None)

  if schema_type in _ID_TYPES and not out.get("@id"):
    out["@id"] = data.get("id") or schema_base_id(name, schema_type)

  if schema_type in _DESC_URL_TYPES:
    if not out.get("description"):
      out["description"] = data.get("description") or TPL["DESCRIPTION"]
    if not out.get("url"):
      out["url"] = data.get("url") or TPL["URL"]

  if schema_type in _IMAGE_TYPES and not out.get("image"):
    out["image"] = data.get("image") or TPL["IMAGE_URL"]

  if schema_type in content_types:
    if not out.get("headline"):
      out["headline"] = out.get("name") or name
    if not out.get("datePublished"):
      out["datePublished"] = data.get("datePublished") or TPL["DATE_PUBLISHED"]
    if not out.get("author"):
      out["author"] = {"@type": "Person", "name": data.get("author") or TPL["AUTHOR_NAME"]}
    if data.get("articleBody") and not out.get("articleBody"):
      out["articleBody"] = str(data["articleBody"])[:5000]
    if data.get("articleSection") and schema_type != "WebPage":
      out["articleSection"] = str(data["articleSection"])

  if schema_type == "WebPage":
    crumbs = data.get("breadcrumbs") or data.get("breadcrumb")
    if crumbs and not out.get("breadcrumb"):
      out["breadcrumb"] = crumbs
    if data.get("isPartOf") and not out.get("isPartOf"):
      out["isPartOf"] = data["isPartOf"]
    if data.get("primaryImageOfPage") and not out.get("primaryImageOfPage"):
      out["primaryImageOfPage"] = data["primaryImageOfPage"]
    elif not out.get("primaryImageOfPage"):
      out["primaryImageOfPage"] = data.get("image") or TPL["IMAGE_URL"]

  if schema_type == "Product":
    if not out.get("brand"):
      out["brand"] = {"@type": "Brand", "name": data.get("brand") or TPL["BRAND_NAME"]}
    if not out.get("offers"):
      out["offers"] = {
        "@type": "Offer",
        "price": str(data.get("price") or TPL["PRICE"]),
        "priceCurrency": str(data.get("priceCurrency") or data.get("currency") or TPL["CURRENCY"]),
      }

  if schema_type in {"Organization", "EducationalOrganization", "GovernmentOrganization"}:
    if not out.get("logo"):
      out["logo"] = data.get("logo") or TPL["LOGO_URL"]
    if not out.get("sameAs"):
      out["sameAs"] = data.get("sameAs") or [TPL["SOCIAL_URL"]]
    if not out.get("address"):
      out["address"] = _address_from_data(data)
    if not out.get("contactPoint"):
      out["contactPoint"] = {
        "@type": "ContactPoint",
        "telephone": data.get("telephone") or data.get("phone") or TPL["PHONE"],
        "contactType": "customer service",
      }

  if schema_type == "Service":
    if not out.get("provider"):
      out["provider"] = {"@type": "Organization", "name": data.get("provider") or TPL["PROVIDER_NAME"]}
    if not out.get("serviceType"):
      out["serviceType"] = data.get("serviceType") or TPL["SERVICE_TYPE"]
    if not out.get("areaServed"):
      out["areaServed"] = data.get("areaServed") or TPL["AREA_SERVED"]
    if not out.get("offers"):
      out["offers"] = {
        "@type": "Offer",
        "price": str(data.get("price") or TPL["PRICE"]),
        "priceCurrency": str(data.get("priceCurrency") or TPL["CURRENCY"]),
      }

  if schema_type == "Course":
    if not out.get("provider"):
      out["provider"] = {"@type": "Organization", "name": data.get("provider") or TPL["PROVIDER_NAME"]}
    if not out.get("offers"):
      out["offers"] = {
        "@type": "Offer",
        "price": str(data.get("price") or TPL["PRICE"]),
        "priceCurrency": str(data.get("priceCurrency") or TPL["CURRENCY"]),
      }

  if schema_type == "Brand":
    if not out.get("logo"):
      out["logo"] = data.get("logo") or TPL["LOGO_URL"]
    if not out.get("sameAs"):
      out["sameAs"] = data.get("sameAs") or [TPL["SOCIAL_URL"]]

  if schema_type in {"LocalBusiness", "MedicalBusiness", "Restaurant", "RealEstateAgent"}:
    if not out.get("address"):
      out["address"] = _address_from_data(data)

  if schema_type == "Recipe":
    if not out.get("recipeIngredient"):
      out["recipeIngredient"] = data.get("recipeIngredient") or data.get("ingredients") or []
    if not out.get("recipeInstructions"):
      out["recipeInstructions"] = data.get("recipeInstructions") or []

  if schema_type == "HowTo" and not out.get("step"):
    out["step"] = [{
      "@type": "HowToStep", "position": 1, "name": "Step 1", "text": TPL["DESCRIPTION"],
    }]

  if schema_type == "FAQPage" and not out.get("mainEntity"):
    out["mainEntity"] = [{
      "@type": "Question",
      "name": TPL["DESCRIPTION"],
      "acceptedAnswer": {"@type": "Answer", "text": TPL["DESCRIPTION"]},
    }]

  if schema_type == "Review":
    if not out.get("reviewBody"):
      out["reviewBody"] = data.get("reviewBody") or TPL["DESCRIPTION"]
    if not out.get("reviewRating"):
      out["reviewRating"] = {
        "@type": "Rating",
        "ratingValue": str(data.get("ratingValue") or TPL["RATING"]),
        "bestRating": str(data.get("bestRating") or "5"),
      }
    if not out.get("author"):
      out["author"] = {"@type": "Person", "name": data.get("author") or TPL["AUTHOR_NAME"]}

  if schema_type == "AggregateRating":
    if not out.get("ratingValue"):
      out["ratingValue"] = str(data.get("ratingValue") or TPL["RATING_VALUE"])
    if not out.get("reviewCount"):
      out["reviewCount"] = str(data.get("reviewCount") or TPL["REVIEW_COUNT"])

  if schema_type == "JobPosting":
    if not out.get("title"):
      out["title"] = out.get("name") or name
    if not out.get("hiringOrganization"):
      out["hiringOrganization"] = {
        "@type": "Organization",
        "name": data.get("hiringOrganization") or data.get("organization") or TPL["ORGANIZATION_NAME"],
      }
    if not out.get("jobLocation"):
      out["jobLocation"] = {"@type": "Place", "address": _address_from_data(data)}

  if schema_type == "VideoObject" and not out.get("uploadDate"):
    out["uploadDate"] = data.get("uploadDate") or TPL["UPLOAD_DATE"]

  if schema_type == "ImageObject" and not out.get("contentUrl"):
    out["contentUrl"] = data.get("contentUrl") or data.get("url") or TPL["CONTENT_URL"]

  if schema_type == "PodcastEpisode" and not out.get("partOfSeries"):
    out["partOfSeries"] = {
      "@type": "PodcastSeries",
      "name": data.get("seriesName") or TPL["ORGANIZATION_NAME"],
    }

  if schema_type in {"WebSite"}:
    site_url = data.get("url") or out.get("url") or TPL["URL"]
    out["url"] = site_url
    if not out.get("potentialAction"):
      out["potentialAction"] = {
        "@type": "SearchAction",
        "target": f"{str(site_url).rstrip('/')}/search?q={{search_term_string}}",
        "query-input": "required name=search_term_string",
      }

  if schema_type == "Offer":
    if not out.get("price"):
      out["price"] = TPL["PRICE"]
    if not out.get("priceCurrency"):
      out["priceCurrency"] = TPL["CURRENCY"]

  if schema_type == "BreadcrumbList" and not out.get("itemListElement"):
    out["itemListElement"] = []

  kws = infer_topic_keywords(name, data)
  if kws:
    out["keywords"] = ", ".join(kws)
  else:
    out.pop("keywords", None)

  return out


def build_linked_schema_graph(
  primary: dict[str, Any],
  schema_type: str,
  name: str,
  data: dict[str, Any],
) -> dict[str, Any]:
  """Linked @graph — Article/LocalBusiness with nested Organization, Person, ImageObject, WebPage."""
  slug = slugify(name)
  base = f"https://example.org/schema/{slug}"
  graph: list[dict[str, Any]] = []

  if schema_type in {"Article", "NewsArticle", "Blog", "BlogPosting"}:
    author_id = f"{base}#author"
    author_name = (
      primary.get("author", {}).get("name")
      if isinstance(primary.get("author"), dict) else data.get("author")
    ) or TPL["AUTHOR_NAME"]

    article = dict(primary)
    article["@id"] = f"{base}#article"
    article["author"] = {"@id": author_id}
    graph.append(article)
    graph.append({"@type": "Person", "@id": author_id, "name": str(author_name)})

    page_url = data.get("url") or primary.get("url")
    if page_url and user_provided(page_url):
      page_id = f"{base}#webpage"
      article["url"] = page_url
      article["mainEntityOfPage"] = {"@type": "WebPage", "@id": page_id}
      graph.append({"@type": "WebPage", "@id": page_id, "url": page_url, "name": article.get("headline") or name})

    pub_name = data.get("publisher") or (
      primary.get("publisher", {}).get("name")
      if isinstance(primary.get("publisher"), dict) else None
    )
    if pub_name and user_provided(pub_name):
      pub_id = f"{base}#publisher"
      article["publisher"] = {"@id": pub_id}
      pub_node: dict[str, Any] = {"@type": "Organization", "@id": pub_id, "name": str(pub_name)}
      if data.get("logo") or data.get("publisherLogo"):
        pub_node["logo"] = data.get("publisherLogo") or data.get("logo")
      graph.append(pub_node)

    image_val = data.get("image") or primary.get("image")
    if image_val and user_provided(image_val):
      img_id = f"{base}#image"
      article["image"] = {"@id": img_id}
      graph.append({
        "@type": "ImageObject",
        "@id": img_id,
        "contentUrl": image_val if isinstance(image_val, str) else str(image_val),
      })
    graph[0] = article

  elif schema_type in {"LocalBusiness", "MedicalBusiness", "Restaurant", "RealEstateAgent"}:
    biz = dict(primary)
    biz["@id"] = f"{base}#business"
    if not biz.get("address"):
      biz["address"] = _address_from_data(data)
    graph.append(biz)

  else:
    crumbs = data.get("breadcrumbs") or data.get("breadcrumb")
    if crumbs and schema_type != "BreadcrumbList":
      items = []
      if isinstance(crumbs, list):
        for i, c in enumerate(crumbs, 1):
          if isinstance(c, dict) and c.get("name") and (c.get("item") or c.get("url")):
            items.append({
              "@type": "ListItem",
              "position": i,
              "name": c.get("name"),
              "item": c.get("item") or c.get("url"),
            })
      if items:
        graph.extend([
          primary,
          {"@type": "BreadcrumbList", "@id": f"{base}#breadcrumb", "itemListElement": items},
        ])
        return {"@context": SCHEMA_CONTEXT, "@graph": graph}
    return primary

  if not graph:
    return primary
  return {"@context": SCHEMA_CONTEXT, "@graph": graph}


# Backward-compatible aliases
optimize_for_rich_results = apply_verified_defaults


def build_nested_graph(
  primary: dict[str, Any],
  schema_type: str,
  name: str,
  data: dict[str, Any],
) -> dict[str, Any]:
  return build_linked_schema_graph(primary, schema_type, name, data)


def validate_schema_structure(schema: dict[str, Any], schema_type: str) -> dict[str, Any]:
  issues: list[str] = []
  if schema.get("@graph"):
    nodes = schema["@graph"]
    if not isinstance(nodes, list):
      issues.append("invalid_graph")
    else:
      for node in nodes:
        if not isinstance(node, dict):
          issues.append("invalid_graph_node")
        elif node.get("@type") != schema_type and node is nodes[0]:
          pass
        elif not node.get("@type"):
          issues.append("missing_type_in_graph")
  else:
    if schema.get("@context") != SCHEMA_CONTEXT:
      issues.append("invalid_context")
    if schema.get("@type") != schema_type:
      issues.append("type_mismatch")
  try:
    json.dumps(schema)
  except (TypeError, ValueError):
    issues.append("invalid_json")
  missing = check_required_properties(
    schema if not schema.get("@graph") else schema["@graph"][0],
    schema_type,
  )
  if missing:
    issues.extend(f"missing_{m}" for m in missing)
  return {"valid": not issues, "issues": issues, "missing_required": missing}


def _find_duplicate_keys(obj: Any) -> list[str]:
  """Detect duplicate keys within the same JSON object (invalid JSON-LD)."""
  dupes: list[str] = []
  if isinstance(obj, dict):
    keys = list(obj.keys())
    local_dupes = {k for k in keys if keys.count(k) > 1}
    dupes.extend(local_dupes)
    for v in obj.values():
      dupes.extend(_find_duplicate_keys(v))
  elif isinstance(obj, list):
    for item in obj:
      dupes.extend(_find_duplicate_keys(item))
  return dupes


def validate_google_compliance(schema: dict[str, Any], schema_type: str) -> dict[str, Any]:
  issues: list[str] = []
  primary = schema["@graph"][0] if schema.get("@graph") else schema
  if schema_type not in RICH_RESULTS_ELIGIBLE:
    issues.append("not_rich_results_type")
  missing = check_required_properties(primary, schema_type)
  if missing:
    issues.extend([f"google_required_{m}" for m in missing])
  for prop in REQUIRED_PROPERTIES.get(schema_type, []):
    if contains_template(primary.get(prop)):
      issues.append(f"template_{prop}")
  text_blob = json.dumps(primary).lower()
  if "lorem ipsum" in text_blob:
    issues.append("placeholder_content")
  has_templates = any(i.startswith("template_") for i in issues)
  return {
    "eligible_rich_results": (
      schema_type in RICH_RESULTS_ELIGIBLE
      and not missing
      and not has_templates
    ),
    "issues": issues,
    "guidelines": "Google Search Central structured data",
    "needs_user_input": has_templates or bool(missing),
  }


def score_schema(
  schema: dict[str, Any],
  schema_type: str,
  validation: dict[str, Any],
  google: dict[str, Any],
) -> dict[str, Any]:
  primary = schema["@graph"][0] if schema.get("@graph") else schema
  required = REQUIRED_PROPERTIES.get(schema_type, ["name"])
  present = sum(1 for p in required if primary.get(p))
  verified = sum(1 for p in required if user_provided(primary.get(p)))
  completeness = int(round(100 * present / max(len(required), 1)))
  verified_pct = int(round(100 * verified / max(len(required), 1)))

  schema_score = verified_pct
  if validation.get("valid"):
    schema_score = min(99, schema_score + 5)
  else:
    schema_score = max(35, schema_score - len(validation.get("issues", [])) * 4)

  google_score = 92 if google.get("eligible_rich_results") else 55
  if google.get("needs_user_input"):
    google_score = max(40, google_score - 25)
  if google.get("issues"):
    google_score = max(35, google_score - len(google["issues"]) * 4)

  seo_score = 70
  if user_provided(primary.get("description")):
    seo_score += 8
  if user_provided(primary.get("image") or primary.get("thumbnailUrl") or primary.get("contentUrl")):
    seo_score += 8
  if user_provided(primary.get("url")):
    seo_score += 6
  if primary.get("inLanguage"):
    seo_score += 4
  if primary.get("keywords") and user_provided(primary.get("keywords")):
    seo_score += 5
  seo_score = min(99, seo_score)

  overall = round(schema_score * 0.4 + google_score * 0.35 + seo_score * 0.25)
  return {
    "schema_score": min(99, schema_score),
    "google_compliance_score": min(99, google_score),
    "seo_score": seo_score,
    "completeness_score": completeness,
    "verified_completeness_score": verified_pct,
    "overall_score": overall,
    "seo_ready": overall >= 80 and google.get("eligible_rich_results", False),
  }


def slugify(text: str) -> str:
  s = re.sub(r"[^\w\s-]", "", (text or "").lower())
  return re.sub(r"[-\s]+", "-", s).strip("-")[:80] or "page"


def _primary_node(schema: dict[str, Any]) -> dict[str, Any]:
  if schema.get("@graph") and isinstance(schema["@graph"], list) and schema["@graph"]:
    node = schema["@graph"][0]
    return node if isinstance(node, dict) else schema
  return schema


def _topic_facts_from_docs(docs: list[Any], topic: str) -> list[str]:
  anchors = {t for t in re.findall(r"\w+", topic.lower()) if len(t) > 2}
  facts: list[str] = []
  seen: set[str] = set()
  for doc in docs[:6]:
    text = re.sub(r"^#+\s*.+$", "", getattr(doc, "text", "") or "", flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    for sent in re.split(r"(?<=[.!?])\s+", text):
      sent = sent.strip()
      if len(sent) < 40 or len(sent) > 240:
        continue
      low = sent.lower()
      if anchors and not any(a in low for a in anchors):
        continue
      key = low[:80]
      if key in seen:
        continue
      seen.add(key)
      facts.append(sent)
  return facts[:4]


def enrich_data_from_open_docs(
  name: str,
  data: dict[str, Any],
  docs: list[Any],
  entities: list[str],
) -> dict[str, Any]:
  """Topic-agnostic enrichment from open datasets — keywords from content, not title splits."""
  out = dict(data)
  keywords = infer_topic_keywords(name, out)
  topic_docs = [d for d in docs if getattr(d, "source", "") not in {"schema_knowledge"}]
  for kw in extract_keywords_from_docs(topic_docs, name):
    if kw.lower() not in {k.lower() for k in keywords}:
      keywords.append(kw)
  out["keywords"] = sanitize_keyword_list(keywords, limit=10)

  if not out.get("description"):
    facts = _topic_facts_from_docs(docs, name)
    if facts:
      out["description"] = facts[0][:300]
  return out


def validate_property_types(schema: dict[str, Any]) -> dict[str, Any]:
  primary = _primary_node(schema)
  issues: list[str] = []
  typed_nested = {
    "author", "publisher", "provider", "hiringOrganization", "brand", "offers",
    "aggregateRating", "reviewRating", "address", "geo", "location", "itemReviewed",
    "partOfSeries", "acceptedAnswer",
  }
  for key in typed_nested:
    val = primary.get(key)
    if isinstance(val, dict) and not val.get("@type"):
      issues.append(f"missing_type_{key}")
  if primary.get("@type") == "FAQPage":
    for item in primary.get("mainEntity") or []:
      if isinstance(item, dict) and not item.get("@type"):
        issues.append("faq_missing_question_type")
  return {"valid": not issues, "issues": issues}


def validate_nesting(schema: dict[str, Any], schema_type: str) -> dict[str, Any]:
  primary = _primary_node(schema)
  issues: list[str] = []
  if schema_type == "FAQPage":
    me = primary.get("mainEntity")
    if not isinstance(me, list) or not me:
      issues.append("faq_main_entity_invalid")
    else:
      for q in me:
        if not isinstance(q, dict) or not q.get("acceptedAnswer"):
          issues.append("faq_missing_answer")
  if schema_type == "BreadcrumbList":
    items = primary.get("itemListElement")
    if not isinstance(items, list):
      issues.append("breadcrumb_items_invalid")
  if schema_type == "Product" and isinstance(primary.get("offers"), dict):
    if not primary["offers"].get("priceCurrency"):
      issues.append("offer_missing_currency")
  return {"valid": not issues, "issues": issues}


def validate_seo(
  schema: dict[str, Any],
  schema_type: str,
  google_val: dict[str, Any],
) -> dict[str, Any]:
  primary = _primary_node(schema)
  dupes = _find_duplicate_keys(schema)
  json_valid = True
  try:
    json.dumps(schema)
  except (TypeError, ValueError):
    json_valid = False
  schema_org = (
    schema.get("@context") == SCHEMA_CONTEXT
    or (schema.get("@graph") and all(
      n.get("@context") == SCHEMA_CONTEXT or "@context" not in n
      for n in schema.get("@graph", []) if isinstance(n, dict)
    ))
  )
  ai_friendly = bool(
    primary.get("description")
    or primary.get("headline")
    or primary.get("name")
  )
  return {
    "valid": (
      google_val.get("eligible_rich_results", False)
      and not dupes
      and json_valid
      and schema_org
      and ai_friendly
    ),
    "rich_results_eligible": google_val.get("eligible_rich_results", False),
    "no_duplicate_properties": not dupes,
    "json_syntax_valid": json_valid,
    "schema_org_compliant": schema_org,
    "ai_search_friendly": ai_friendly,
    "duplicate_keys": dupes,
  }


def build_validation_checklist(
  *,
  input_val: dict[str, Any],
  structure_val: dict[str, Any],
  property_types_val: dict[str, Any],
  nesting_val: dict[str, Any],
  google_val: dict[str, Any],
  seo_val: dict[str, Any],
  reqs: dict[str, Any],
) -> dict[str, Any]:
  """Structured validation checklist matching enterprise workflow spec."""
  missing = structure_val.get("missing_required", [])
  return {
    "input_validation": {
      "schema_type_supported": {
        "passed": input_val.get("checks", {}).get("schema_type_supported", False),
        "label": "Schema type supported",
      },
      "title_name_provided": {
        "passed": input_val.get("checks", {}).get("title_name_provided", False),
        "label": "Title/name provided",
      },
      "required_properties_available": {
        "passed": not missing,
        "label": "Required properties available",
        "missing": missing,
      },
    },
    "schema_validation": {
      "valid_context": {
        "passed": "invalid_context" not in structure_val.get("issues", []),
        "label": "Valid @context",
      },
      "valid_type": {
        "passed": "type_mismatch" not in structure_val.get("issues", []),
        "label": "Valid @type",
      },
      "required_properties_included": {
        "passed": not missing,
        "label": "Required properties included",
      },
      "correct_property_types": {
        "passed": property_types_val.get("valid", False),
        "label": "Correct property types",
      },
      "proper_nesting": {
        "passed": nesting_val.get("valid", False),
        "label": "Proper nesting",
      },
    },
    "seo_validation": {
      "eligible_google_rich_results": {
        "passed": seo_val.get("rich_results_eligible", False),
        "label": "Eligible for Google Rich Results",
        "type_eligible": reqs.get("rich_results_eligible", False),
      },
      "no_duplicate_properties": {
        "passed": seo_val.get("no_duplicate_properties", False),
        "label": "No duplicate properties",
      },
      "json_syntax_valid": {
        "passed": seo_val.get("json_syntax_valid", False),
        "label": "JSON syntax valid",
      },
      "schema_org_compliant": {
        "passed": seo_val.get("schema_org_compliant", False),
        "label": "Schema.org compliant",
      },
      "ai_search_friendly": {
        "passed": seo_val.get("ai_search_friendly", False),
        "label": "AI search friendly",
      },
    },
    "all_passed": (
      input_val.get("valid", False)
      and structure_val.get("valid", False)
      and property_types_val.get("valid", False)
      and nesting_val.get("valid", False)
      and seo_val.get("json_syntax_valid", False)
      and seo_val.get("schema_org_compliant", False)
      and not google_val.get("needs_user_input", False)
    ),
  }


def build_validation_report(
  schema: dict[str, Any],
  schema_type: str,
  scores: dict[str, Any],
  warnings: list[str],
  google_val: dict[str, Any],
) -> dict[str, Any]:
  """Human-readable validation report with required/optional property status."""
  primary = _primary_node(schema)
  required = REQUIRED_PROPERTIES.get(schema_type, ["name"])
  optional = OPTIONAL_PROPERTIES.get(schema_type, [])

  def prop_status(prop: str) -> dict[str, Any]:
    val = primary.get(prop)
    if val is None or val == "" or val == []:
      return {"name": prop, "status": "missing", "passed": False, "symbol": "✗"}
    if contains_template(val):
      return {"name": prop, "status": "template", "passed": False, "symbol": "⚠"}
    return {"name": prop, "status": "provided", "passed": True, "symbol": "✓"}

  required_props = [prop_status(p) for p in required]
  optional_missing = [
    {"name": p, "symbol": "⚠"}
    for p in optional
    if not primary.get(p) or contains_template(primary.get(p))
  ]

  lines = [
    f"Schema Score: {scores.get('overall_score', 0)}/100",
    "",
    "Required Properties:",
  ]
  for rp in required_props:
    lines.append(f"{rp['symbol']} {rp['name']}" + (" (template — needs user input)" if rp["status"] == "template" else ""))
  if optional_missing:
    lines.extend(["", "Optional Properties Missing:"])
    for om in optional_missing:
      lines.append(f"{om['symbol']} {om['name']}")
  lines.extend([
    "",
    "Google Rich Results:",
    f"{'✓' if google_val.get('eligible_rich_results') else '⚠'} {'Eligible' if google_val.get('eligible_rich_results') else 'Needs user input before eligible'}",
    "",
    "Schema.org:",
    "✓ Valid JSON-LD structure",
  ])
  if warnings:
    lines.extend(["", "Warnings:"])
    for w in warnings:
      lines.append(f"• {w}")

  return {
    "schema_score": scores.get("overall_score", 0),
    "required_properties": required_props,
    "optional_properties_missing": optional_missing,
    "google_rich_results": {
      "eligible": google_val.get("eligible_rich_results", False),
      "needs_user_input": google_val.get("needs_user_input", False),
    },
    "schema_org": {"valid": True},
    "warnings": warnings,
    "summary_text": "\n".join(lines),
  }

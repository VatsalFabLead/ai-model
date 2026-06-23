"""Advanced Schema.org JSON-LD engine.

Worldwide categories, multilingual defaults, SEO enrichment, and a dedicated
schema knowledge base — 100% custom, no GPT/Claude/Gemini.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMA_KB_PATH = PROJECT_ROOT / "data" / "schema_knowledge.jsonl"

_SCHEMA_TYPES = [
  # Content & publishing
  "Article", "NewsArticle", "Blog", "WebPage", "WebSite", "FAQPage", "HowTo",
  "Review", "AggregateRating", "BreadcrumbList", "SitelinksSearchBox",
  # E-commerce & products
  "Product", "Offer", "Brand", "Store",
  # Business & organizations
  "Organization", "LocalBusiness", "Service", "MedicalBusiness", "Restaurant",
  "RealEstateAgent", "EducationalOrganization", "GovernmentOrganization",
  "FinancialService", "LegalService", "Dentist", "Hotel",
  # People & media
  "Person", "Event", "VideoObject", "ImageObject", "PodcastEpisode",
  "Book", "Movie", "MusicRecording", "SoftwareApplication", "MobileApplication",
  # Education & jobs
  "Course", "JobPosting", "Recipe", "TouristAttraction",
]

_CATEGORIES: dict[str, dict[str, Any]] = {
  "content": {
    "label": "Content & Publishing",
    "description": "Articles, blogs, news, FAQs, and how-to guides",
    "types": [
      "Article", "NewsArticle", "Blog", "WebPage", "FAQPage", "HowTo",
      "Review", "AggregateRating", "BreadcrumbList",
    ],
  },
  "ecommerce": {
    "label": "E-commerce & Products",
    "description": "Products, offers, brands, and online stores",
    "types": ["Product", "Offer", "Brand", "Store"],
  },
  "business": {
    "label": "Business & Services",
    "description": "Companies, local businesses, and professional services",
    "types": [
      "Organization", "LocalBusiness", "Service", "MedicalBusiness",
      "Restaurant", "RealEstateAgent", "FinancialService", "LegalService",
      "Dentist", "Hotel",
    ],
  },
  "education": {
    "label": "Education & Careers",
    "description": "Courses, schools, and job listings",
    "types": ["Course", "JobPosting", "EducationalOrganization"],
  },
  "media": {
    "label": "Media & Entertainment",
    "description": "Video, audio, books, movies, and tourism",
    "types": [
      "VideoObject", "ImageObject", "PodcastEpisode", "Book", "Movie",
      "MusicRecording", "TouristAttraction", "Event",
    ],
  },
  "technology": {
    "label": "Technology & Apps",
    "description": "Software, mobile apps, and websites",
    "types": ["SoftwareApplication", "MobileApplication", "WebSite", "SitelinksSearchBox"],
  },
  "people": {
    "label": "People & Profiles",
    "description": "Individual profiles and personal branding",
    "types": ["Person"],
  },
  "food": {
    "label": "Food & Recipes",
    "description": "Recipes and restaurant listings",
    "types": ["Recipe", "Restaurant"],
  },
}

_TYPE_MAP = {t.lower(): t for t in _SCHEMA_TYPES}
_TYPE_TO_CATEGORY: dict[str, str] = {}
for cat_id, cat in _CATEGORIES.items():
  for t in cat["types"]:
    _TYPE_TO_CATEGORY.setdefault(t, cat_id)

_SCHEMA_HELP: dict[str, str] = {
  "Article": "Blog posts, guides, and editorial content",
  "NewsArticle": "Breaking news and journalism",
  "Blog": "Blog homepage or blog section",
  "WebPage": "Any standard web page",
  "WebSite": "Entire website metadata",
  "FAQPage": "Frequently asked questions page",
  "HowTo": "Step-by-step tutorials and guides",
  "Review": "Single product or service review",
  "AggregateRating": "Average star ratings summary",
  "BreadcrumbList": "Navigation breadcrumb trail",
  "SitelinksSearchBox": "Google sitelinks search box",
  "Product": "E-commerce product pages",
  "Offer": "Standalone price/offer block",
  "Brand": "Brand identity and logo",
  "Store": "Online or physical retail store",
  "Organization": "Company or NGO profile",
  "LocalBusiness": "Local shop, clinic, or office",
  "Service": "Professional service offering",
  "MedicalBusiness": "Clinic, hospital, or healthcare",
  "Restaurant": "Restaurant business details",
  "RealEstateAgent": "Real estate agency profile",
  "EducationalOrganization": "School, college, or academy",
  "GovernmentOrganization": "Government body or agency",
  "FinancialService": "Bank, insurance, or fintech",
  "LegalService": "Law firm or legal consultancy",
  "Dentist": "Dental clinic or practice",
  "Hotel": "Hotel, resort, or lodging",
  "Person": "Individual professional profile",
  "Event": "Conferences, webinars, concerts",
  "VideoObject": "Video content pages",
  "ImageObject": "Image gallery or asset page",
  "PodcastEpisode": "Podcast episode details",
  "Book": "Book product or author page",
  "Movie": "Film or cinema content",
  "MusicRecording": "Song or album track",
  "SoftwareApplication": "Desktop or web software",
  "MobileApplication": "iOS/Android app pages",
  "Course": "Online or offline course",
  "JobPosting": "Job vacancy listing",
  "Recipe": "Cooking recipe with ingredients",
  "TouristAttraction": "Landmarks, museums, parks",
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en",
  "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es",
  "french": "fr", "fr": "fr",
  "german": "de", "de": "de",
  "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar",
  "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh", "mandarin": "zh",
  "korean": "ko", "ko": "ko",
  "italian": "it", "it": "it",
  "russian": "ru", "ru": "ru",
  "turkish": "tr", "tr": "tr",
  "indonesian": "id", "id": "id",
  "bengali": "bn", "bn": "bn",
  "tamil": "ta", "ta": "ta",
  "marathi": "mr", "mr": "mr",
  "urdu": "ur", "ur": "ur",
  "vietnamese": "vi", "vi": "vi",
  "thai": "th", "th": "th",
  "dutch": "nl", "nl": "nl",
  "polish": "pl", "pl": "pl",
}

# Localized publisher / org fallback labels (aesthetic, worldwide).
_LOCALE_LABELS: dict[str, dict[str, str]] = {
  "en": {"publisher": "Editorial Team", "org": "Your Organization", "author": "Content Author"},
  "hi": {"publisher": "संपादकीय टीम", "org": "आपका संगठन", "author": "लेखक"},
  "es": {"publisher": "Equipo Editorial", "org": "Su Organización", "author": "Autor del Contenido"},
  "fr": {"publisher": "Équipe Éditoriale", "org": "Votre Organisation", "author": "Auteur"},
  "de": {"publisher": "Redaktion", "org": "Ihre Organisation", "author": "Autor"},
  "pt": {"publisher": "Equipe Editorial", "org": "Sua Organização", "author": "Autor"},
  "ar": {"publisher": "فريق التحرير", "org": "مؤسستك", "author": "المؤلف"},
  "ja": {"publisher": "編集チーム", "org": "あなたの組織", "author": "著者"},
  "zh": {"publisher": "编辑团队", "org": "您的组织", "author": "作者"},
  "ko": {"publisher": "편집팀", "org": "귀하의 조직", "author": "작성자"},
  "it": {"publisher": "Team Editoriale", "org": "La Tua Organizzazione", "author": "Autore"},
  "ru": {"publisher": "Редакция", "org": "Ваша организация", "author": "Автор"},
}

# Recommended fields per type for quality scoring.
_RECOMMENDED: dict[str, list[str]] = {
  "Article": ["headline", "description", "image", "author", "publisher", "datePublished", "inLanguage"],
  "Product": ["description", "image", "brand", "offers", "sku", "aggregateRating"],
  "FAQPage": ["mainEntity"],
  "LocalBusiness": ["address", "telephone", "openingHours", "geo", "image"],
  "JobPosting": ["description", "datePosted", "employmentType", "hiringOrganization", "jobLocation"],
  "Recipe": ["recipeIngredient", "recipeInstructions", "cookTime", "image"],
  "Event": ["startDate", "location", "description", "image"],
  "Course": ["description", "provider", "offers"],
  "Hotel": ["address", "telephone", "starRating", "amenityFeature"],
  "SoftwareApplication": ["applicationCategory", "operatingSystem", "offers"],
}

_schema_kb: KnowledgeBase | None = None


def get_schema_kb() -> KnowledgeBase:
  global _schema_kb
  if _schema_kb is None:
    _schema_kb = load_knowledge_base(knowledge_path=SCHEMA_KB_PATH)
  return _schema_kb


def reload_schema_kb() -> None:
  global _schema_kb
  _schema_kb = load_knowledge_base(knowledge_path=SCHEMA_KB_PATH)


def supported_types() -> list[dict[str, str]]:
  return [
    {
      "type": t,
      "help": _SCHEMA_HELP.get(t, ""),
      "category": _TYPE_TO_CATEGORY.get(t, "content"),
    }
    for t in _SCHEMA_TYPES
  ]


def supported_categories() -> list[dict[str, Any]]:
  out = []
  for cat_id, cat in _CATEGORIES.items():
    out.append({
      "id": cat_id,
      "label": cat["label"],
      "description": cat["description"],
      "types": cat["types"],
      "type_count": len(cat["types"]),
    })
  return out


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"},
    {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"},
    {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"},
    {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"},
    {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"},
    {"name": "Korean", "code": "ko"},
    {"name": "Italian", "code": "it"},
    {"name": "Russian", "code": "ru"},
    {"name": "Turkish", "code": "tr"},
    {"name": "Indonesian", "code": "id"},
    {"name": "Bengali", "code": "bn"},
    {"name": "Tamil", "code": "ta"},
    {"name": "Marathi", "code": "mr"},
    {"name": "Urdu", "code": "ur"},
    {"name": "Vietnamese", "code": "vi"},
    {"name": "Thai", "code": "th"},
    {"name": "Dutch", "code": "nl"},
    {"name": "Polish", "code": "pl"},
  ]


def norm_type(schema_type: str) -> str:
  key = (schema_type or "").strip().lower()
  if key in _TYPE_MAP:
    return _TYPE_MAP[key]
  raise ValueError(f"Unsupported schema type: {schema_type}")


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def locale_labels(language: str | None) -> dict[str, str]:
  code = bcp47(language)
  return _LOCALE_LABELS.get(code, _LOCALE_LABELS["en"])


def get_guidance(schema_type: str, language: str | None = None) -> str:
  """Retrieve schema best-practice guidance from the training knowledge base."""
  kb = get_schema_kb()
  lang = bcp47(language)
  queries = [
    f"{schema_type} schema.org JSON-LD best practices",
    f"{schema_type} required fields SEO",
    f"schema markup {schema_type} multilingual {lang}",
  ]
  chunks: list[str] = []
  for q in queries:
    answer, score = kb.search(q)
    if answer and score > 0.05 and answer not in chunks:
      chunks.append(answer)
  return "\n\n".join(chunks[:2])


def quality_report(schema: dict[str, Any], schema_type: str) -> dict[str, Any]:
  recommended = _RECOMMENDED.get(schema_type, ["name", "description", "url"])
  missing = [f for f in recommended if not schema.get(f)]
  present = len(recommended) - len(missing)
  score = int(round(100 * present / max(len(recommended), 1)))
  return {
    "completeness_score": score,
    "seo_ready": score >= 70,
    "missing_recommended_fields": missing,
    "field_count": len(schema),
  }


def sanitize_schema(obj: dict[str, Any], schema_type: str) -> dict[str, Any]:
  """Ensure valid core structure and strip empty values."""
  obj["@context"] = "https://schema.org"
  obj["@type"] = schema_type

  def _strip_empty(d: Any) -> Any:
    if isinstance(d, dict):
      cleaned = {k: _strip_empty(v) for k, v in d.items() if v is not None and v != "" and v != []}
      return cleaned
    if isinstance(d, list):
      return [_strip_empty(x) for x in d if x is not None and x != ""]
    return d

  return _strip_empty(obj)


def apply_global_enrichment(
  schema: dict[str, Any],
  schema_type: str,
  name: str,
  data: dict[str, Any],
  language: str | None,
) -> dict[str, Any]:
  """Worldwide SEO enrichment: language, aesthetics, accessibility."""
  lang = bcp47(language)
  labels = locale_labels(language)

  if schema_type in {
    "Article", "NewsArticle", "Blog", "WebPage", "HowTo", "FAQPage",
    "Product", "Course", "Recipe", "Event", "VideoObject", "Book", "Movie",
  }:
    schema.setdefault("inLanguage", lang)

  if schema_type in {"Article", "NewsArticle", "Blog", "WebPage"}:
    schema.setdefault("isAccessibleForFree", True)
    if not schema.get("author"):
      schema["author"] = {"@type": "Person", "name": labels["author"]}
    if not schema.get("publisher"):
      schema["publisher"] = {"@type": "Organization", "name": labels["publisher"]}

  if schema_type == "WebSite" and not schema.get("potentialAction"):
    site_url = str(data.get("url") or schema.get("url") or "https://example.com")
    schema["potentialAction"] = {
      "@type": "SearchAction",
      "target": f"{site_url.rstrip('/')}/search?q={{search_term_string}}",
      "query-input": "required name=search_term_string",
    }

  if schema_type == "Organization" and not schema.get("logo") and data.get("logo"):
    schema["logo"] = str(data["logo"])

  keywords = data.get("keywords")
  if keywords and "keywords" not in schema:
    if isinstance(keywords, list):
      schema["keywords"] = ", ".join(str(k) for k in keywords)
    else:
      schema["keywords"] = str(keywords)

  return sanitize_schema(schema, schema_type)


def pretty_jsonld(schema: dict[str, Any]) -> str:
  return json.dumps(schema, indent=2, ensure_ascii=False)


def category_for_type(schema_type: str) -> str:
  return _TYPE_TO_CATEGORY.get(schema_type, "content")

"""Schema Markup Generator (JSON-LD) — advanced, multilingual, worldwide.

Uses deterministic rich templates + dedicated schema knowledge base (RAG) +
optional enhancement via your local custom model (Qwen/llama.cpp/Ollama/custom).
No GPT, Claude, Gemini, or proprietary APIs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.engine import schema_engine
from app.services.provider_base import ModelProvider


def supported_types() -> list[dict[str, str]]:
  return schema_engine.supported_types()


def supported_categories() -> list[dict[str, Any]]:
  return schema_engine.supported_categories()


def supported_languages() -> list[dict[str, str]]:
  return schema_engine.supported_languages()


def _clean_text(value: str | None) -> str:
  return re.sub(r"\s+", " ", (value or "").strip())


def _pick(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
  for k in keys:
    if k in data and data[k] is not None and data[k] != "":
      return data[k]
  return default


def _iso_now() -> str:
  return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _postal_address(data: dict[str, Any]) -> dict[str, Any]:
  return {
    "@type": "PostalAddress",
    "streetAddress": str(_pick(data, "streetAddress", "address", default="")),
    "addressLocality": str(_pick(data, "city", "addressLocality", default="")),
    "addressRegion": str(_pick(data, "state", "addressRegion", default="")),
    "postalCode": str(_pick(data, "postalCode", "zip", default="")),
    "addressCountry": str(_pick(data, "country", "addressCountry", default="")),
  }


def _base(schema_type: str, name: str, data: dict[str, Any], language: str | None) -> dict[str, Any]:
  labels = schema_engine.locale_labels(language)
  obj: dict[str, Any] = {
    "@context": "https://schema.org",
    "@type": schema_type,
    "name": _clean_text(name) or _clean_text(str(_pick(data, "name", "title", default="Untitled"))),
  }
  if _pick(data, "description"):
    obj["description"] = _clean_text(str(_pick(data, "description")))
  if _pick(data, "url"):
    obj["url"] = str(_pick(data, "url"))
  if _pick(data, "image"):
    obj["image"] = _pick(data, "image")
  if _pick(data, "id"):
    obj["@id"] = str(_pick(data, "id"))
  return obj


def _content_fields(s: dict[str, Any], data: dict[str, Any], language: str | None) -> None:
  labels = schema_engine.locale_labels(language)
  s["headline"] = _clean_text(str(_pick(data, "headline", "title", default=s["name"])))
  s["datePublished"] = str(_pick(data, "datePublished", default=_iso_now()))
  s["dateModified"] = str(_pick(data, "dateModified", default=s["datePublished"]))
  author = _pick(data, "author", default=labels["author"])
  s["author"] = {"@type": "Person", "name": str(author)} if isinstance(author, str) else author
  publisher = _pick(data, "publisher", default=labels["publisher"])
  s["publisher"] = (
    {"@type": "Organization", "name": str(publisher)}
    if isinstance(publisher, str)
    else publisher
  )
  if _pick(data, "articleSection"):
    s["articleSection"] = str(_pick(data, "articleSection"))


def _business_fields(s: dict[str, Any], data: dict[str, Any]) -> None:
  s["address"] = _postal_address(data)
  if _pick(data, "telephone", "phone"):
    s["telephone"] = str(_pick(data, "telephone", "phone"))
  if _pick(data, "openingHours"):
    s["openingHours"] = _pick(data, "openingHours")
  if _pick(data, "priceRange"):
    s["priceRange"] = str(_pick(data, "priceRange"))
  if _pick(data, "latitude") and _pick(data, "longitude"):
    s["geo"] = {
      "@type": "GeoCoordinates",
      "latitude": float(_pick(data, "latitude")),
      "longitude": float(_pick(data, "longitude")),
    }


def _build(schema_type: str, name: str, data: dict[str, Any], language: str | None) -> dict[str, Any]:
  s = _base(schema_type, name, data, language)

  if schema_type in {"Article", "NewsArticle", "Blog", "WebPage"}:
    _content_fields(s, data, language)

  elif schema_type == "Product":
    s["brand"] = {"@type": "Brand", "name": str(_pick(data, "brand", default="Generic"))}
    if _pick(data, "sku"):
      s["sku"] = str(_pick(data, "sku"))
    if _pick(data, "gtin"):
      s["gtin"] = str(_pick(data, "gtin"))
    price = _pick(data, "price", default="0")
    currency = _pick(data, "priceCurrency", "currency", default="USD")
    s["offers"] = {
      "@type": "Offer",
      "price": str(price),
      "priceCurrency": str(currency),
      "availability": str(_pick(data, "availability", default="https://schema.org/InStock")),
      "url": str(_pick(data, "url", default="https://example.com")),
    }
    if _pick(data, "ratingValue"):
      s["aggregateRating"] = {
        "@type": "AggregateRating",
        "ratingValue": str(_pick(data, "ratingValue")),
        "reviewCount": str(_pick(data, "reviewCount", default="1")),
        "bestRating": str(_pick(data, "bestRating", default="5")),
      }

  elif schema_type in {
    "LocalBusiness", "MedicalBusiness", "Restaurant", "RealEstateAgent",
    "Dentist", "Hotel", "Store", "FinancialService", "LegalService",
  }:
    _business_fields(s, data)
    if schema_type == "Restaurant" and _pick(data, "servesCuisine"):
      s["servesCuisine"] = _pick(data, "servesCuisine")
    if schema_type == "Hotel" and _pick(data, "starRating"):
      s["starRating"] = {
        "@type": "Rating",
        "ratingValue": str(_pick(data, "starRating")),
        "bestRating": "5",
      }
    if schema_type in {"FinancialService", "LegalService"} and _pick(data, "serviceType"):
      s["serviceType"] = str(_pick(data, "serviceType"))

  elif schema_type == "FAQPage":
    faqs = _pick(data, "faqs", "faq", default=[])
    entities = []
    for item in faqs if isinstance(faqs, list) else []:
      q = _clean_text(str(_pick(item, "question", "q", default="")))
      a = _clean_text(str(_pick(item, "answer", "a", default="")))
      if q and a:
        entities.append({
          "@type": "Question",
          "name": q,
          "acceptedAnswer": {"@type": "Answer", "text": a},
        })
    s["mainEntity"] = entities

  elif schema_type in {"Organization", "EducationalOrganization", "GovernmentOrganization"}:
    if _pick(data, "logo"):
      s["logo"] = str(_pick(data, "logo"))
    if _pick(data, "sameAs"):
      s["sameAs"] = _pick(data, "sameAs")
    if _pick(data, "foundingDate"):
      s["foundingDate"] = str(_pick(data, "foundingDate"))

  elif schema_type == "Recipe":
    s["recipeIngredient"] = _pick(data, "recipeIngredient", "ingredients", default=[])
    instructions = _pick(data, "recipeInstructions", "instructions", default=[])
    if isinstance(instructions, list) and instructions and isinstance(instructions[0], str):
      s["recipeInstructions"] = [
        {"@type": "HowToStep", "text": _clean_text(str(step))} for step in instructions
      ]
    else:
      s["recipeInstructions"] = instructions
    for time_key in ("prepTime", "cookTime", "totalTime"):
      if _pick(data, time_key):
        s[time_key] = str(_pick(data, time_key))
    if _pick(data, "recipeCuisine"):
      s["recipeCuisine"] = str(_pick(data, "recipeCuisine"))

  elif schema_type == "Event":
    s["startDate"] = str(_pick(data, "startDate", default=_iso_now()))
    s["endDate"] = str(_pick(data, "endDate", default=s["startDate"]))
    s["eventAttendanceMode"] = str(
      _pick(data, "eventAttendanceMode", default="https://schema.org/OnlineEventAttendanceMode")
    )
    s["eventStatus"] = str(_pick(data, "eventStatus", default="https://schema.org/EventScheduled"))
    if _pick(data, "location"):
      loc = _pick(data, "location")
      s["location"] = (
        {"@type": "Place", "name": str(loc)} if isinstance(loc, str) else loc
      )

  elif schema_type == "Person":
    if _pick(data, "jobTitle"):
      s["jobTitle"] = str(_pick(data, "jobTitle"))
    if _pick(data, "sameAs"):
      s["sameAs"] = _pick(data, "sameAs")
    if _pick(data, "worksFor"):
      s["worksFor"] = {"@type": "Organization", "name": str(_pick(data, "worksFor"))}

  elif schema_type == "WebSite":
    s["url"] = str(_pick(data, "url", default="https://example.com"))

  elif schema_type == "HowTo":
    steps = _pick(data, "steps", default=[])
    s["step"] = [
      {
        "@type": "HowToStep",
        "position": idx + 1,
        "name": _clean_text(str(_pick(st, "name", default=f"Step {idx + 1}"))),
        "text": _clean_text(str(_pick(st, "text", default=""))),
      }
      for idx, st in enumerate(steps)
      if isinstance(st, dict)
    ]
    if _pick(data, "totalTime"):
      s["totalTime"] = str(_pick(data, "totalTime"))

  elif schema_type == "Review":
    s["reviewRating"] = {
      "@type": "Rating",
      "ratingValue": str(_pick(data, "ratingValue", default="5")),
      "bestRating": str(_pick(data, "bestRating", default="5")),
    }
    s["author"] = {"@type": "Person", "name": str(_pick(data, "author", default="Anonymous"))}
    if _pick(data, "reviewBody"):
      s["reviewBody"] = str(_pick(data, "reviewBody"))
    if _pick(data, "itemReviewed"):
      s["itemReviewed"] = {"@type": "Product", "name": str(_pick(data, "itemReviewed"))}

  elif schema_type == "AggregateRating":
    s["ratingValue"] = str(_pick(data, "ratingValue", default="4.5"))
    s["reviewCount"] = str(_pick(data, "reviewCount", default="1"))
    s["bestRating"] = str(_pick(data, "bestRating", default="5"))

  elif schema_type == "Service":
    s["provider"] = {"@type": "Organization", "name": str(_pick(data, "provider", default="Your Organization"))}
    if _pick(data, "serviceType"):
      s["serviceType"] = str(_pick(data, "serviceType"))
    if _pick(data, "areaServed"):
      s["areaServed"] = _pick(data, "areaServed")

  elif schema_type == "Course":
    s["provider"] = {
      "@type": "Organization",
      "name": str(_pick(data, "provider", default="Your Academy")),
      "sameAs": str(_pick(data, "providerUrl", default="https://example.com")),
    }
    if _pick(data, "educationalLevel"):
      s["educationalLevel"] = str(_pick(data, "educationalLevel"))

  elif schema_type == "JobPosting":
    s["title"] = s["name"]
    s["description"] = str(_pick(data, "description", default=s.get("description", "")))
    s["datePosted"] = str(_pick(data, "datePosted", default=_iso_now()))
    s["employmentType"] = str(_pick(data, "employmentType", default="FULL_TIME"))
    s["hiringOrganization"] = {
      "@type": "Organization",
      "name": str(_pick(data, "hiringOrganization", "organization", default="Your Company")),
      "sameAs": str(_pick(data, "organizationUrl", default="https://example.com")),
    }
    s["jobLocation"] = {
      "@type": "Place",
      "address": _postal_address(data),
    }
    if _pick(data, "baseSalary"):
      s["baseSalary"] = _pick(data, "baseSalary")

  elif schema_type == "VideoObject":
    s["uploadDate"] = str(_pick(data, "uploadDate", default=_iso_now()))
    s["thumbnailUrl"] = _pick(data, "thumbnailUrl", "image", default=[])
    if _pick(data, "duration"):
      s["duration"] = str(_pick(data, "duration"))
    if _pick(data, "contentUrl"):
      s["contentUrl"] = str(_pick(data, "contentUrl"))

  elif schema_type == "ImageObject":
    s["contentUrl"] = str(_pick(data, "contentUrl", "url", default="https://example.com/image.jpg"))
    if _pick(data, "caption"):
      s["caption"] = str(_pick(data, "caption"))

  elif schema_type == "PodcastEpisode":
    s["partOfSeries"] = {
      "@type": "PodcastSeries",
      "name": str(_pick(data, "seriesName", default="Podcast Series")),
    }
    s["datePublished"] = str(_pick(data, "datePublished", default=_iso_now()))

  elif schema_type == "Offer":
    s["price"] = str(_pick(data, "price", default="0"))
    s["priceCurrency"] = str(_pick(data, "priceCurrency", "currency", default="USD"))
    s["availability"] = str(_pick(data, "availability", default="https://schema.org/InStock"))

  elif schema_type == "Brand":
    if _pick(data, "logo"):
      s["logo"] = str(_pick(data, "logo"))

  elif schema_type == "BreadcrumbList":
    crumbs = _pick(data, "breadcrumbs", "items", default=[])
    s["itemListElement"] = []
    pos = 1
    for c in crumbs if isinstance(crumbs, list) else []:
      cname = str(_pick(c, "name", default=""))
      curl = str(_pick(c, "item", "url", default=""))
      if cname and curl:
        s["itemListElement"].append({
          "@type": "ListItem",
          "position": pos,
          "name": cname,
          "item": curl,
        })
        pos += 1

  elif schema_type == "SitelinksSearchBox":
    site_url = str(_pick(data, "url", default="https://example.com"))
    target = str(_pick(data, "target", default=f"{site_url}/search?q={{search_term_string}}"))
    s["url"] = site_url
    s["potentialAction"] = {
      "@type": "SearchAction",
      "target": target,
      "query-input": "required name=search_term_string",
    }

  elif schema_type in {"SoftwareApplication", "MobileApplication"}:
    s["applicationCategory"] = str(_pick(data, "applicationCategory", default="BusinessApplication"))
    s["operatingSystem"] = str(_pick(data, "operatingSystem", default="Web"))
    if _pick(data, "softwareVersion"):
      s["softwareVersion"] = str(_pick(data, "softwareVersion"))
    s["offers"] = {
      "@type": "Offer",
      "price": str(_pick(data, "price", default="0")),
      "priceCurrency": str(_pick(data, "priceCurrency", "currency", default="USD")),
    }
    if _pick(data, "downloadUrl"):
      s["downloadUrl"] = str(_pick(data, "downloadUrl"))

  elif schema_type == "Book":
    if _pick(data, "author"):
      s["author"] = {"@type": "Person", "name": str(_pick(data, "author"))}
    if _pick(data, "isbn"):
      s["isbn"] = str(_pick(data, "isbn"))
    if _pick(data, "bookFormat"):
      s["bookFormat"] = str(_pick(data, "bookFormat"))
    s["datePublished"] = str(_pick(data, "datePublished", default=_iso_now()))

  elif schema_type == "Movie":
    if _pick(data, "director"):
      s["director"] = {"@type": "Person", "name": str(_pick(data, "director"))}
    if _pick(data, "genre"):
      s["genre"] = _pick(data, "genre")
    if _pick(data, "duration"):
      s["duration"] = str(_pick(data, "duration"))
    s["datePublished"] = str(_pick(data, "datePublished", default=_iso_now()))

  elif schema_type == "MusicRecording":
    if _pick(data, "byArtist"):
      s["byArtist"] = {"@type": "MusicGroup", "name": str(_pick(data, "byArtist"))}
    if _pick(data, "duration"):
      s["duration"] = str(_pick(data, "duration"))

  elif schema_type == "TouristAttraction":
    _business_fields(s, data)
    if _pick(data, "isAccessibleForFree") is not None:
      s["isAccessibleForFree"] = bool(_pick(data, "isAccessibleForFree"))

  extras = _pick(data, "extra", default={})
  if isinstance(extras, dict):
    for k, v in extras.items():
      if k not in {"@context", "@type"} and v is not None:
        s[k] = v

  return schema_engine.apply_global_enrichment(s, schema_type, name, data, language)


def _extract_json(text: str) -> dict[str, Any] | None:
  raw = (text or "").strip()
  if raw.startswith("```"):
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
  try:
    obj = json.loads(raw)
    return obj if isinstance(obj, dict) else None
  except Exception:
    return None


async def _ai_enhance(
  provider: ModelProvider,
  schema: dict[str, Any],
  schema_type: str,
  language: str | None,
) -> dict[str, Any]:
  lang_code = schema_engine.bcp47(language)
  guidance = schema_engine.get_guidance(schema_type, language)
  lang_line = f"Human-readable text must be in language code: {lang_code}." if language else ""
  system_prompt = (
    "You are an expert Schema.org JSON-LD engineer for worldwide SEO. Improve the "
    "given JSON-LD for richness, aesthetics, and search quality while staying valid. "
    "Rules: keep @context https://schema.org, keep @type unchanged, never invent fake "
    "reviews/ratings/URLs, only add fields justified by existing data, return ONLY a "
    "valid JSON object with no markdown. " + lang_line
  )
  if guidance:
    system_prompt += "\n\nTraining knowledge:\n" + guidance[:2000]

  user_prompt = "Optimize this JSON-LD:\n" + json.dumps(schema, ensure_ascii=False)
  try:
    raw = await provider.chat(
      [{"role": "user", "content": user_prompt}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=900,
      temperature=0.25,
    )
    obj = _extract_json(raw)
    if not obj:
      return schema
    return schema_engine.sanitize_schema(obj, schema_type)
  except Exception:
    return schema


async def generate_schema_markup(
  provider: ModelProvider,
  *,
  schema_type: str,
  name: str,
  data: dict[str, Any] | None = None,
  language: str | None = None,
  ai_enhance: bool = True,
) -> dict[str, Any]:
  data = data or {}
  stype = schema_engine.norm_type(schema_type)
  schema = _build(stype, name, data, language)

  if ai_enhance:
    schema = await _ai_enhance(provider, schema, stype, language)

  quality = schema_engine.quality_report(schema, stype)
  return {
    "schema_type": stype,
    "category": schema_engine.category_for_type(stype),
    "language": schema_engine.bcp47(language),
    "jsonld": schema,
    "jsonld_string": schema_engine.pretty_jsonld(schema),
    "quality": quality,
  }

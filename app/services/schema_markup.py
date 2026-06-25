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
from app.engine.schema_markup_enrichment import TPL, _address_from_data
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


def _postal_address(data: dict[str, Any]) -> dict[str, Any] | None:
  if not any(_pick(data, k) for k in (
    "streetAddress", "address", "city", "addressLocality", "state",
    "addressRegion", "postalCode", "zip", "country", "addressCountry",
  )):
    return None
  return _address_from_data(data)


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


def _webpage_fields(s: dict[str, Any], data: dict[str, Any]) -> None:
  crumbs = _pick(data, "breadcrumbs", "breadcrumb")
  if crumbs:
    s["breadcrumb"] = crumbs
  if _pick(data, "isPartOf"):
    s["isPartOf"] = _pick(data, "isPartOf")
  if _pick(data, "primaryImageOfPage"):
    s["primaryImageOfPage"] = _pick(data, "primaryImageOfPage")
  elif _pick(data, "image"):
    s["primaryImageOfPage"] = _pick(data, "image")


def _content_fields(s: dict[str, Any], data: dict[str, Any], language: str | None) -> None:
  s["headline"] = _clean_text(str(_pick(data, "headline", "title", default=s["name"])))
  if _pick(data, "datePublished"):
    s["datePublished"] = str(_pick(data, "datePublished"))
  if _pick(data, "dateModified"):
    s["dateModified"] = str(_pick(data, "dateModified"))
  author = _pick(data, "author")
  if author:
    s["author"] = {"@type": "Person", "name": str(author)} if isinstance(author, str) else author
  publisher = _pick(data, "publisher")
  if publisher:
    s["publisher"] = (
      {"@type": "Organization", "name": str(publisher)}
      if isinstance(publisher, str)
      else publisher
    )
  if _pick(data, "articleSection"):
    s["articleSection"] = str(_pick(data, "articleSection"))
  if _pick(data, "wordCount"):
    s["wordCount"] = int(_pick(data, "wordCount"))
  if _pick(data, "articleBody"):
    s["articleBody"] = str(_pick(data, "articleBody"))


def _business_fields(s: dict[str, Any], data: dict[str, Any]) -> None:
  addr = _postal_address(data)
  if addr:
    s["address"] = addr
  if _pick(data, "telephone", "phone"):
    s["telephone"] = str(_pick(data, "telephone", "phone"))
  if _pick(data, "openingHours"):
    s["openingHours"] = _pick(data, "openingHours")
  if _pick(data, "priceRange"):
    s["priceRange"] = str(_pick(data, "priceRange"))
  if _pick(data, "url"):
    s["url"] = str(_pick(data, "url"))
  if _pick(data, "logo"):
    s["logo"] = str(_pick(data, "logo"))
  if _pick(data, "sameAs"):
    s["sameAs"] = _pick(data, "sameAs")
  if _pick(data, "latitude") and _pick(data, "longitude"):
    s["geo"] = {
      "@type": "GeoCoordinates",
      "latitude": float(_pick(data, "latitude")),
      "longitude": float(_pick(data, "longitude")),
    }
  if _pick(data, "telephone", "phone") or _pick(data, "contactPoint"):
    s["contactPoint"] = _pick(data, "contactPoint") or {
      "@type": "ContactPoint",
      "telephone": str(_pick(data, "telephone", "phone")),
      "contactType": str(_pick(data, "contactType", default="customer service")),
    }


def _build(schema_type: str, name: str, data: dict[str, Any], language: str | None) -> dict[str, Any]:
  s = _base(schema_type, name, data, language)

  if schema_type in {"Article", "NewsArticle", "Blog", "BlogPosting"}:
    _content_fields(s, data, language)

  elif schema_type == "WebPage":
    _webpage_fields(s, data)

  elif schema_type == "Product":
    if _pick(data, "brand"):
      s["brand"] = {"@type": "Brand", "name": str(_pick(data, "brand"))}
    if _pick(data, "sku"):
      s["sku"] = str(_pick(data, "sku"))
    if _pick(data, "gtin"):
      s["gtin"] = str(_pick(data, "gtin"))
    if _pick(data, "price") or _pick(data, "priceCurrency", "currency"):
      s["offers"] = {
        "@type": "Offer",
        "price": str(_pick(data, "price", default=TPL["PRICE"])),
        "priceCurrency": str(_pick(data, "priceCurrency", "currency", default=TPL["CURRENCY"])),
        "availability": str(_pick(data, "availability", default="https://schema.org/InStock")),
      }
      if _pick(data, "url"):
        s["offers"]["url"] = str(_pick(data, "url"))
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
    addr = _postal_address(data)
    if addr:
      s["address"] = addr
    if _pick(data, "logo"):
      s["logo"] = str(_pick(data, "logo"))
    if _pick(data, "sameAs"):
      s["sameAs"] = _pick(data, "sameAs")
    if _pick(data, "foundingDate"):
      s["foundingDate"] = str(_pick(data, "foundingDate"))
    if _pick(data, "telephone", "phone") or _pick(data, "contactPoint"):
      s["contactPoint"] = _pick(data, "contactPoint") or {
        "@type": "ContactPoint",
        "telephone": str(_pick(data, "telephone", "phone")),
        "contactType": str(_pick(data, "contactType", default="customer service")),
      }

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
    if _pick(data, "startDate"):
      s["startDate"] = str(_pick(data, "startDate"))
    if _pick(data, "endDate"):
      s["endDate"] = str(_pick(data, "endDate"))
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
    if _pick(data, "url"):
      s["url"] = str(_pick(data, "url"))
    site_url = _pick(data, "url") or s.get("url")
    if site_url:
      s["potentialAction"] = {
        "@type": "SearchAction",
        "target": f"{str(site_url).rstrip('/')}/search?q={{search_term_string}}",
        "query-input": "required name=search_term_string",
      }

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
    if _pick(data, "ratingValue"):
      s["reviewRating"] = {
        "@type": "Rating",
        "ratingValue": str(_pick(data, "ratingValue")),
        "bestRating": str(_pick(data, "bestRating", default="5")),
      }
    if _pick(data, "author"):
      s["author"] = {"@type": "Person", "name": str(_pick(data, "author"))}
    if _pick(data, "reviewBody"):
      s["reviewBody"] = str(_pick(data, "reviewBody"))
    if _pick(data, "itemReviewed"):
      s["itemReviewed"] = {"@type": "Product", "name": str(_pick(data, "itemReviewed"))}

  elif schema_type == "AggregateRating":
    if _pick(data, "ratingValue"):
      s["ratingValue"] = str(_pick(data, "ratingValue"))
    if _pick(data, "reviewCount"):
      s["reviewCount"] = str(_pick(data, "reviewCount"))
    if _pick(data, "bestRating"):
      s["bestRating"] = str(_pick(data, "bestRating"))

  elif schema_type == "Service":
    if _pick(data, "provider"):
      s["provider"] = {"@type": "Organization", "name": str(_pick(data, "provider"))}
    if _pick(data, "serviceType"):
      s["serviceType"] = str(_pick(data, "serviceType"))
    if _pick(data, "areaServed"):
      s["areaServed"] = _pick(data, "areaServed")

  elif schema_type == "Course":
    if _pick(data, "provider"):
      s["provider"] = {
        "@type": "Organization",
        "name": str(_pick(data, "provider")),
      }
      if _pick(data, "providerUrl"):
        s["provider"]["sameAs"] = str(_pick(data, "providerUrl"))
    if _pick(data, "educationalLevel"):
      s["educationalLevel"] = str(_pick(data, "educationalLevel"))

  elif schema_type == "JobPosting":
    s["title"] = s["name"]
    if _pick(data, "description"):
      s["description"] = str(_pick(data, "description"))
    if _pick(data, "datePosted"):
      s["datePosted"] = str(_pick(data, "datePosted"))
    if _pick(data, "employmentType"):
      s["employmentType"] = str(_pick(data, "employmentType"))
    if _pick(data, "hiringOrganization", "organization"):
      s["hiringOrganization"] = {
        "@type": "Organization",
        "name": str(_pick(data, "hiringOrganization", "organization")),
      }
      if _pick(data, "organizationUrl"):
        s["hiringOrganization"]["sameAs"] = str(_pick(data, "organizationUrl"))
    addr = _postal_address(data)
    if addr:
      s["jobLocation"] = {"@type": "Place", "address": addr}
    if _pick(data, "baseSalary"):
      s["baseSalary"] = _pick(data, "baseSalary")

  elif schema_type == "VideoObject":
    if _pick(data, "uploadDate"):
      s["uploadDate"] = str(_pick(data, "uploadDate"))
    if _pick(data, "thumbnailUrl", "image"):
      s["thumbnailUrl"] = _pick(data, "thumbnailUrl", "image")
    if _pick(data, "duration"):
      s["duration"] = str(_pick(data, "duration"))
    if _pick(data, "contentUrl"):
      s["contentUrl"] = str(_pick(data, "contentUrl"))

  elif schema_type == "ImageObject":
    if _pick(data, "contentUrl", "url"):
      s["contentUrl"] = str(_pick(data, "contentUrl", "url"))
    if _pick(data, "caption"):
      s["caption"] = str(_pick(data, "caption"))

  elif schema_type == "PodcastEpisode":
    if _pick(data, "seriesName"):
      s["partOfSeries"] = {
        "@type": "PodcastSeries",
        "name": str(_pick(data, "seriesName")),
      }
    if _pick(data, "datePublished"):
      s["datePublished"] = str(_pick(data, "datePublished"))

  elif schema_type == "Offer":
    if _pick(data, "price"):
      s["price"] = str(_pick(data, "price"))
    if _pick(data, "priceCurrency", "currency"):
      s["priceCurrency"] = str(_pick(data, "priceCurrency", "currency"))
    s["availability"] = str(_pick(data, "availability", default="https://schema.org/InStock"))

  elif schema_type == "Brand":
    if _pick(data, "logo"):
      s["logo"] = str(_pick(data, "logo"))
    if _pick(data, "sameAs"):
      s["sameAs"] = _pick(data, "sameAs")

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

  elif schema_type in {"SoftwareApplication", "MobileApplication"}:
    if _pick(data, "applicationCategory"):
      s["applicationCategory"] = str(_pick(data, "applicationCategory"))
    if _pick(data, "operatingSystem"):
      s["operatingSystem"] = str(_pick(data, "operatingSystem"))
    if _pick(data, "softwareVersion"):
      s["softwareVersion"] = str(_pick(data, "softwareVersion"))
    if _pick(data, "price") or _pick(data, "priceCurrency", "currency"):
      s["offers"] = {
        "@type": "Offer",
        "price": str(_pick(data, "price", default=TPL["PRICE"])),
        "priceCurrency": str(_pick(data, "priceCurrency", "currency", default=TPL["CURRENCY"])),
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
    if _pick(data, "datePublished"):
      s["datePublished"] = str(_pick(data, "datePublished"))

  elif schema_type == "Movie":
    if _pick(data, "director"):
      s["director"] = {"@type": "Person", "name": str(_pick(data, "director"))}
    if _pick(data, "genre"):
      s["genre"] = _pick(data, "genre")
    if _pick(data, "duration"):
      s["duration"] = str(_pick(data, "duration"))
    if _pick(data, "datePublished"):
      s["datePublished"] = str(_pick(data, "datePublished"))

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
  ai_enhance: bool = False,
  use_rag: bool = False,
) -> dict[str, Any]:
  from app.engine.schema_markup_rag_pipeline import run_schema_markup_pipeline

  return await run_schema_markup_pipeline(
    schema_type=schema_type,
    name=name,
    data=data,
    language=language,
    ai_enhance=ai_enhance and provider is not None,
    use_rag=use_rag,
    provider=provider,
  )

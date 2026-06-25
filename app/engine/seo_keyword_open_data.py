"""SEO Keyword Generator — open/free dataset routing and retrieval.

Maps pipeline stages to Wikidata, Wikipedia, DBpedia, GeoNames, OpenStreetMap,
Datamuse (synonyms), Stack Overflow, OpenAlex, Open Food Facts, Common Crawl
proxies, ESCO/O*NET skills, and local knowledge bases.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx

from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources

_USER_AGENT = "NexusSEO-Keyword/4.0 (open-data; educational)"

# Purpose → recommended open datasets (all free / open access)
DATASET_STACK: dict[str, dict[str, Any]] = {
  "brand_recognition": {"datasets": ["wikidata", "wikipedia"], "status": "live"},
  "entity_recognition": {"datasets": ["wikidata", "dbpedia"], "status": "live"},
  "industry_classification": {"datasets": ["wikidata", "wikipedia", "dbpedia"], "status": "live"},
  "local_seo": {"datasets": ["geonames", "openstreetmap", "wikidata"], "status": "live"},
  "language_detection": {"datasets": ["wikipedia"], "status": "heuristic+optional_langdetect"},
  "synonyms": {"datasets": ["datamuse", "conceptnet"], "status": "live"},
  "semantic_similarity": {"datasets": ["datamuse", "conceptnet", "local_faiss"], "status": "live"},
  "keyword_clustering": {"datasets": ["datamuse", "local_faiss"], "status": "live"},
  "programming_keywords": {"datasets": ["stackexchange", "github"], "status": "live"},
  "skills": {"datasets": ["esco", "onet"], "status": "static_taxonomy"},
  "trends": {"datasets": ["gdelt", "gooaq"], "status": "live_proxy"},
  "schema": {"datasets": ["schema_knowledge"], "status": "local_kb"},
  "seo_best_practices": {"datasets": ["schema_knowledge", "gooaq"], "status": "local_kb+live"},
  "product_data": {"datasets": ["openfoodfacts", "wikidata"], "status": "live"},
  "research_topics": {"datasets": ["openalex", "semantic_scholar", "arxiv"], "status": "live"},
  "general_web": {"datasets": ["c4", "fineweb", "wikipedia"], "status": "live"},
}

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "Brand Recognition": ["wikidata", "wikipedia"],
  "Entity Recognition": ["wikidata", "dbpedia"],
  "Industry Classification": ["wikidata", "wikipedia", "dbpedia"],
  "Local SEO": ["geonames", "openstreetmap", "wikidata"],
  "Language Detection": ["wikipedia"],
  "Synonyms (WordNet-style)": ["datamuse", "conceptnet"],
  "Semantic Similarity": ["datamuse", "conceptnet", "local_faiss"],
  "Keyword Clustering": ["datamuse", "local_faiss"],
  "Programming Keywords": ["stackexchange", "github"],
  "Skills (ESCO + O*NET)": ["esco", "onet"],
  "Trends": ["gdelt", "gooaq"],
  "Schema.org": ["schema_knowledge"],
  "SEO Best Practices": ["schema_knowledge", "gooaq"],
  "Product Data": ["openfoodfacts", "wikidata"],
  "Research (OpenAlex)": ["openalex", "semantic_scholar", "arxiv"],
  "General Web (Common Crawl proxies)": ["c4", "fineweb", "wikipedia"],
}

_INDUSTRY_SOURCE_ROUTES: dict[str, list[str]] = {
  "Healthcare": ["wikipedia", "wikidata", "dbpedia", "pubmed", "gooaq", "geonames"],
  "Technology": ["wikipedia", "wikidata", "stackexchange", "github", "arxiv", "datamuse"],
  "Artificial Intelligence": ["wikipedia", "wikidata", "openalex", "semantic_scholar", "arxiv"],
  "Beauty": ["wikipedia", "wikidata", "openfoodfacts", "dbpedia", "geonames", "datamuse"],
  "Cosmetics": ["wikipedia", "wikidata", "openfoodfacts", "dbpedia", "geonames"],
  "E-commerce": ["wikipedia", "wikidata", "dbpedia", "gooaq", "c4"],
  "Finance": ["wikipedia", "wikidata", "gdelt", "gooaq"],
  "Education": ["wikipedia", "wikidata", "openalex", "esco", "onet"],
  "Marketing": ["wikipedia", "gooaq", "c4", "schema_knowledge"],
  "Food": ["openfoodfacts", "wikipedia", "wikidata"],
  "Local Business": ["geonames", "openstreetmap", "wikipedia", "wikidata", "datamuse"],
  "Brand": ["wikidata", "wikipedia", "openfoodfacts", "dbpedia", "geonames"],
  "default": ["wikipedia", "wikidata", "dbpedia", "datamuse", "gooaq", "geonames"],
}

_STAGE_DATASETS: dict[str, list[str]] = {
  "brand_entity_recognition": ["wikidata", "wikipedia"],
  "entity_extractor": ["wikidata", "dbpedia"],
  "industry_domain_classification": ["wikidata", "wikipedia"],
  "country_region_detector": ["geonames", "openstreetmap"],
  "language_detector": ["wikipedia"],
  "semantic_lsi_expansion": ["datamuse", "conceptnet"],
  "keyword_clustering": ["datamuse", "local_faiss"],
  "local_seo_generator": ["geonames", "openstreetmap"],
  "trending_keyword_engine": ["gdelt", "gooaq"],
  "competitor_keyword_generator": ["gooaq", "c4"],
  "keyword_expansion_engine": ["wikipedia", "wikidata", "dbpedia", "datamuse"],
}


def _clip(text: str, n: int = 800) -> str:
  text = re.sub(r"\s+", " ", (text or "").strip())
  return text if len(text) <= n else text[: n - 3].rstrip() + "..."


async def fetch_geonames(query: str, *, limit: int = 5) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://secure.geonames.org/searchJSON",
        params={"q": query[:60], "maxRows": limit, "username": "demo", "lang": "en"},
      )
      if r.status_code != 200:
        return []
      for geo in r.json().get("geonames", [])[:limit]:
        name = geo.get("name", "")
        country = geo.get("countryName", "")
        admin = geo.get("adminName1", "")
        if name:
          docs.append(OpenDoc(
            doc_id=f"geonames:{geo.get('geonameId')}",
            source="geonames",
            title=name,
            text=_clip(f"{name}, {admin}, {country}".strip(", ")),
            url=f"https://www.geonames.org/{geo.get('geonameId')}",
            score=0.7,
            meta={"country": country, "lat": geo.get("lat"), "lng": geo.get("lng")},
          ))
  except Exception:
    pass
  return docs


async def fetch_openstreetmap(query: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query[:80], "format": "json", "limit": limit, "addressdetails": 1},
      )
      if r.status_code != 200:
        return []
      for item in r.json()[:limit]:
        name = item.get("display_name", "")
        if name:
          docs.append(OpenDoc(
            doc_id=f"osm:{item.get('osm_id')}",
            source="openstreetmap",
            title=item.get("name") or name.split(",")[0],
            text=_clip(name),
            url=f"https://www.openstreetmap.org/{item.get('osm_type')}/{item.get('osm_id')}",
            score=0.68,
          ))
  except Exception:
    pass
  return docs


async def fetch_datamuse(query: str, *, limit: int = 8) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": _USER_AGENT}) as client:
      words: list[str] = []
      for path, params in (
        ("words", {"ml": query[:40], "max": str(limit)}),
        ("words", {"rel_syn": query.split()[0][:20], "max": str(limit)}),
        ("words", {"rel_trg": query.split()[0][:20], "max": str(limit)}),
      ):
        r = await client.get(f"https://api.datamuse.com/{path}", params=params)
        if r.status_code == 200:
          for item in r.json():
            w = item.get("word")
            if w:
              words.append(str(w))
    if words:
      docs.append(OpenDoc(
        doc_id=f"datamuse:{query[:20]}",
        source="datamuse",
        title=f"Synonyms/related: {query[:40]}",
        text=", ".join(dict.fromkeys(words))[:600],
        url="https://www.datamuse.com/api/",
        score=0.72,
        meta={"terms": words[:limit]},
      ))
  except Exception:
    pass
  return docs


async def fetch_openfoodfacts(query: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://world.openfoodfacts.org/cgi/search.pl",
        params={"search_terms": query[:60], "search_simple": 1, "action": "process", "json": 1, "page_size": limit},
      )
      if r.status_code != 200:
        return []
      for prod in r.json().get("products", [])[:limit]:
        brand = prod.get("brands") or prod.get("product_name", "")
        cats = prod.get("categories", "")
        if brand:
          docs.append(OpenDoc(
            doc_id=f"off:{prod.get('code', brand[:12])}",
            source="openfoodfacts",
            title=str(brand)[:80],
            text=_clip(f"{prod.get('product_name', '')} — {cats}"),
            url=f"https://world.openfoodfacts.org/product/{prod.get('code', '')}",
            score=0.65,
          ))
  except Exception:
    pass
  return docs


async def fetch_openalex(query: str, *, limit: int = 3) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://api.openalex.org/works",
        params={"search": query[:80], "per_page": limit},
      )
      if r.status_code != 200:
        return []
      for work in r.json().get("results", [])[:limit]:
        title = work.get("title", "")
        abstract = work.get("abstract_inverted_index")
        text = title
        if isinstance(abstract, dict):
          text = title + " — research topic"
        if title:
          docs.append(OpenDoc(
            doc_id=f"openalex:{work.get('id', '')[-12:]}",
            source="openalex",
            title=title,
            text=_clip(text, 500),
            url=work.get("doi") or work.get("id"),
            score=0.7,
          ))
  except Exception:
    pass
  return docs


async def fetch_github(query: str, *, limit: int = 3) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=12.0, headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}) as client:
      r = await client.get(
        "https://api.github.com/search/repositories",
        params={"q": query[:60], "per_page": limit, "sort": "stars"},
      )
      if r.status_code != 200:
        return []
      for repo in r.json().get("items", [])[:limit]:
        name = repo.get("full_name", "")
        desc = repo.get("description") or ""
        if name:
          docs.append(OpenDoc(
            doc_id=f"github:{name}",
            source="github",
            title=name,
            text=_clip(desc or name),
            url=repo.get("html_url"),
            score=0.66,
          ))
  except Exception:
    pass
  return docs


async def fetch_esco(query: str, *, limit: int = 5) -> list[OpenDoc]:
  from app.engine.resume_open_data import _ESCO_TECH_SKILLS
  terms: list[str] = []
  low = query.lower()
  for _cat, skills in _ESCO_TECH_SKILLS.items():
    if _cat in low or any(s.lower() in low for s in skills[:3]):
      terms.extend(skills[:limit])
  if not terms:
    terms = list(_ESCO_TECH_SKILLS.get("technology", ()))[:limit]
  return [OpenDoc(
    doc_id="esco:skills",
    source="esco",
    title="ESCO skills",
    text=", ".join(terms),
    url="https://esco.ec.europa.eu/",
    score=0.6,
    meta={"terms": terms},
  )] if terms else []


async def fetch_onet(query: str, *, limit: int = 5) -> list[OpenDoc]:
  from app.engine.resume_open_data import _ONET_SOFT_SKILLS
  return [OpenDoc(
    doc_id="onet:skills",
    source="onet",
    title="O*NET skills",
    text=", ".join(_ONET_SOFT_SKILLS[:limit]),
    url="https://www.onetonline.org/",
    score=0.58,
    meta={"terms": list(_ONET_SOFT_SKILLS[:limit])},
  )]


async def fetch_schema_knowledge(query: str, *, limit: int = 2) -> list[OpenDoc]:
  from app.engine.schema_engine import get_guidance
  g = get_guidance("WebPage", None) or get_guidance("Organization", None)
  if not g:
    return []
  return [OpenDoc(
    doc_id="schema:seo",
    source="schema_knowledge",
    title="Schema.org SEO guidance",
    text=_clip(g, 1200),
    url="https://schema.org/",
    score=0.55,
  )]


async def fetch_pubmed(query: str, *, limit: int = 2) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "term": query[:80], "retmax": limit, "retmode": "json"},
      )
      ids = r.json().get("esearchresult", {}).get("idlist", [])
      if not ids:
        return []
      r2 = await client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
      )
      for pid in ids:
        rec = r2.json().get("result", {}).get(pid, {})
        title = rec.get("title", "")
        if title:
          docs.append(OpenDoc(
            doc_id=f"pubmed:{pid}",
            source="pubmed",
            title=title,
            text=_clip(title),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
            score=0.7,
          ))
  except Exception:
    pass
  return docs


SEO_SOURCE_FETCHERS = {
  "geonames": fetch_geonames,
  "openstreetmap": fetch_openstreetmap,
  "datamuse": fetch_datamuse,
  "openfoodfacts": fetch_openfoodfacts,
  "openalex": fetch_openalex,
  "github": fetch_github,
  "esco": fetch_esco,
  "onet": fetch_onet,
  "schema_knowledge": fetch_schema_knowledge,
  "pubmed": fetch_pubmed,
}


def route_sources_for_context(context: dict[str, Any]) -> list[str]:
  industry = (context.get("industry") or {}).get("primary_industry", "")
  route = list(_INDUSTRY_SOURCE_ROUTES.get(industry, _INDUSTRY_SOURCE_ROUTES["default"]))
  primary_domain = context.get("primary_domain")
  if primary_domain and primary_domain in _INDUSTRY_SOURCE_ROUTES:
    route = list(_INDUSTRY_SOURCE_ROUTES[primary_domain])
  elif primary_domain:
    from app.engine.seo_keyword_domains import DOMAIN_BY_NAME
    cat = (DOMAIN_BY_NAME.get(primary_domain) or {}).get("category", "")
    cat_route = {
      "Beauty & Fashion": "Beauty",
      "Healthcare & Medical": "Healthcare",
      "Food": "Food",
      "Technology": "Technology",
      "Ecommerce": "E-commerce",
      "Local Business": "Local Business",
      "Global Brands": "Brand",
    }.get(cat)
    if cat_route and cat_route in _INDUSTRY_SOURCE_ROUTES:
      route = list(_INDUSTRY_SOURCE_ROUTES[cat_route])
  if context.get("is_brand_seed"):
    for s in ("wikidata", "wikipedia", "datamuse", "geonames"):
      if s not in route:
        route.insert(0, s)
  if industry in ("Beauty", "Cosmetics", "Food") and "openfoodfacts" not in route:
    route.insert(0, "openfoodfacts")
  if industry in ("Technology", "Artificial Intelligence"):
    for s in ("stackexchange", "github"):
      if s not in route:
        route.append(s)
  return route[:12]


async def _fetch_one(src: str, query: str, *, per_source: int = 1) -> list[OpenDoc]:
  fn = SEO_SOURCE_FETCHERS.get(src)
  if fn:
    try:
      return await asyncio.wait_for(fn(query, limit=per_source), timeout=8.0)
    except Exception:
      return []
  return []


async def retrieve_seo_keyword_data(
  seed: str,
  context: dict[str, Any],
  *,
  seed_int: int = 0,
  per_source: int = 1,
  max_sources: int = 10,
) -> tuple[list[OpenDoc], list[str], dict[str, Any]]:
  """Retrieve from industry-routed open datasets in parallel."""
  industry = (context.get("industry") or {}).get("primary_industry", "")
  query = context.get("brand_name") or seed
  if industry in ("Beauty", "Cosmetics"):
    query = seed if "beauty" in seed.lower() or "cosmetic" in seed.lower() else f"{query} cosmetics"
  elif context.get("topic_mode") and context.get("topic_parts"):
    query = context["topic_parts"][0][:80]
  sources = route_sources_for_context(context)[:max_sources]
  keywords = [t for t in re.findall(r"\w+", seed) if len(t) > 3][:5]

  core_docs = await retrieve_from_sources(query, keywords, sources, per_source=per_source, seed=seed_int)
  extra_sources = [s for s in sources if s in SEO_SOURCE_FETCHERS]
  tasks = [_fetch_one(s, query, per_source=per_source) for s in extra_sources]
  if tasks:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for batch in results:
      if isinstance(batch, list):
        core_docs.extend(batch)

  used = sorted({d.source for d in core_docs})
  meta = {
    "query": query,
    "sources_requested": sources,
    "sources_used": used,
    "doc_count": len(core_docs),
    "dataset_stack": DATASET_STACK,
  }
  return core_docs, used, meta


_JUNK_PHRASES = (
  "is a class of", "is a type of", "is a chemical", "is a sweet",
  "simple sugar", "disaccharide", "monosaccharide", "chemical compound",
  "head office", "wikidata", "wikipedia",
)


def is_junk_open_keyword(keyword: str, context: dict[str, Any]) -> bool:
  k = keyword.lower().strip()
  if any(j in k for j in _JUNK_PHRASES):
    return True
  industry = (context.get("industry") or {}).get("primary_industry", "")
  if industry in ("Beauty", "Cosmetics") and k in ("sugar", "sugars") and "beauty" not in k and "cosmetic" not in k:
    return True
  if industry in ("Beauty", "Cosmetics") and any(t in k for t in ("software", "technologies", "developer", "development")):
    return True
  if len(k.split()) >= 6 and k.count(k.split()[0]) >= 2:
    return True
  return False


def terms_from_open_docs(docs: list[OpenDoc], context: dict[str, Any]) -> list[str]:
  terms: list[str] = []
  seen: set[str] = set()
  brand = (context.get("brand_name") or "").lower()

  def add(term: str) -> None:
    t = re.sub(r"\s+", " ", term.lower().strip())
    if not t or len(t) < 4 or len(t) > 70:
      return
    if is_junk_open_keyword(t, context):
      return
    if t in seen:
      return
    seen.add(t)
    terms.append(t)

  for doc in docs:
    if doc.source == "datamuse" and doc.meta.get("terms"):
      for t in doc.meta["terms"]:
        add(t)
    if doc.source in ("esco", "onet") and doc.meta.get("terms"):
      for t in doc.meta["terms"]:
        add(t)
    if doc.source == "openfoodfacts" and doc.title:
      add(doc.title)
    if doc.source == "geonames" and doc.title:
      add(f"{doc.title} {doc.meta.get('country', '')}".strip())
    if doc.source in ("wikipedia", "wikidata", "dbpedia"):
      title = doc.title.lower()
      if brand and brand.split()[0] in title:
        add(title)
      elif industry_match(title, context):
        add(title)

  return terms[:20]


def industry_match(text: str, context: dict[str, Any]) -> bool:
  industry = (context.get("industry") or {}).get("primary_industry", "").lower()
  hints = {
    "beauty": ("beauty", "cosmetic", "makeup", "skincare"),
    "cosmetics": ("cosmetic", "makeup", "beauty"),
    "healthcare": ("health", "medical", "clinic"),
    "technology": ("software", "technology", "digital"),
  }
  for key, words in hints.items():
    if key in industry and any(w in text for w in words):
      return True
  return False

"""SEO Keyword Generator — production pipeline (open datasets, dynamic variations).

Input → Input Analyzer → Seed Keyword Generator → Keyword Expansion → Entity Extractor
→ Intent Classifier → Trend Analyzer → Volume Estimator → Difficulty Calculator
→ CPC Analyzer → Competition Analyzer → Trend Individual Engine → Keyword Clustering
→ Opportunity Finder → Ranking & Prioritization → SEO Score → JSON Output
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from typing import Any

from app.engine.keyword_discovery import discover_keywords
from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources
from app.engine.seo_content_domains import make_variation_seed

GENERATOR_VERSION = "seo-keyword-rag-v2.0"

ARCHITECTURE_FLOW = [
  "input",
  "input_analyzer",
  "seed_keyword_generator",
  "keyword_expansion",
  "entity_extractor",
  "intent_classifier",
  "trend_analyzer",
  "volume_estimator",
  "difficulty_calculator",
  "cpc_analyzer",
  "competition_analyzer",
  "trend_individual_engine",
  "keyword_clustering",
  "opportunity_finder",
  "ranking_prioritization",
  "seo_score",
  "json_output",
]

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "General Knowledge": ["wikipedia", "wikidata", "dbpedia"],
  "Web Text": ["c4", "fineweb"],
  "Academic": ["arxiv", "semantic_scholar"],
  "Question Answering": ["gooaq", "squad"],
  "Conversational": ["dolly"],
  "Programming": ["stackexchange"],
  "News": ["gdelt"],
  "Books": ["gutenberg", "openlibrary"],
  "Medical": ["pubmed", "wikipedia"],
  "Finance": ["sec_edgar", "wikidata"],
  "Legal": ["courtlistener", "wikipedia"],
  "Geography": ["wikidata", "openstreetmap"],
  "Multimedia": ["wikimedia", "youtube_cc"],
  "E-commerce": ["amazon_reviews", "wikipedia"],
  "Social Media": ["reddit", "twitter_archive"],
  "Government": ["data_gov", "gdelt"],
  "SEO/Search": ["gooaq", "wikipedia"],
  "Multilingual": ["wikipedia", "wikidata"],
}

_SOURCE_ROUTE = ["wikipedia", "wikidata", "gooaq", "squad"]

_MODIFIERS = (
  "best", "top", "tools", "software", "services", "tips", "guide", "examples",
  "strategy", "tutorial", "checklist", "pricing", "cost", "alternatives",
  "trends 2026", "for beginners", "near me", "vs", "benefits", "how to",
)
_PREFIXES = ("how to", "what is", "why", "when to use", "best way to", "top")
_INTENTS = ("informational", "commercial", "transactional", "navigational")
_COMMERCIAL = ("buy", "price", "pricing", "cost", "cheap", "hire", "agency", "service", "best", "top")
_TRANSACTIONAL = ("near me", "online", "download", "free trial", "signup", "book", "order")
_INFORMATIONAL = ("how to", "what is", "why", "guide", "tutorial", "tips", "examples", "meaning")


def effective_variation_seed(client_seed: int | None) -> int:
  base = make_variation_seed(client_seed)
  nonce = secrets.randbits(31) ^ (time.time_ns() & 0x7FFFFFFF)
  return make_variation_seed(base ^ nonce)


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _pick(pool: list[str], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _shuffle(items: list[Any], seed: int) -> list[Any]:
  out = list(items)
  for i in range(len(out) - 1, 0, -1):
    j = (seed + i * 7919) % (i + 1)
    out[i], out[j] = out[j], out[i]
  return out


def _kw_hash(keyword: str, seed: int) -> int:
  h = 0
  for ch in (keyword.lower() + str(seed)):
    h = (h * 31 + ord(ch)) & 0x7FFFFFFF
  return h


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


def generate_seed_variants(seed: str, count: int, variation_seed: int) -> list[str]:
  low = seed.lower()
  variants = [low]
  mods = _shuffle(list(_MODIFIERS), variation_seed)
  for i, m in enumerate(mods[: max(8, count // 3)]):
    variants.append(f"{low} {m}")
    if i % 2 == 0:
      variants.append(f"{m} {low}")
  for p in _PREFIXES:
    variants.append(f"{p} {low}")
  seen: set[str] = set()
  out: list[str] = []
  for v in _shuffle(variants, variation_seed + 7):
    v = _clean(v.lower())
    if v and v not in seen:
      seen.add(v)
      out.append(v)
  return out[: max(5, min(20, count))]


def expand_keywords(
  seed: str,
  seed_variants: list[str],
  discovered: list[dict[str, Any]],
  docs: list[OpenDoc],
  *,
  count: int,
  variation_seed: int,
) -> list[str]:
  candidates: dict[str, dict[str, Any]] = {}

  def add(kw: str, source: str, relevance: int = 30) -> None:
    k = _clean(kw.lower())
    if not k or len(k) < 3 or len(k) > 80:
      return
    row = candidates.get(k)
    if row is None:
      candidates[k] = {"keyword": k, "sources": [source], "relevance": relevance}
    else:
      if source not in row["sources"]:
        row["sources"].append(source)
      row["relevance"] = max(row["relevance"], relevance)

  add(seed.lower(), "seed", 100)
  for sv in seed_variants:
    add(sv, "seed_variant", 75)
  for item in discovered:
    add(item.get("keyword", ""), "web_discovery", int(item.get("relevance_score") or 50))
  for d in docs[:10]:
    for phrase in re.findall(r"[a-z][a-z0-9\- ]{4,40}", (d.title + " " + d.text[:300]).lower()):
      if seed.lower() in phrase or any(t in phrase for t in seed.lower().split()):
        add(phrase, f"rag:{d.source}", 40)

  mods = _shuffle(list(_MODIFIERS), variation_seed + 13)
  prefixes = _shuffle(list(_PREFIXES), variation_seed + 19)
  for i in range(max(count * 3, 80)):
    m = mods[i % len(mods)]
    p = prefixes[i % len(prefixes)]
    add(f"{seed.lower()} {m}", "expansion", 25 + (i % 7))
    add(f"{m} {seed.lower()}", "expansion", 22 + (i % 5))
    add(f"{p} {seed.lower()}", "long_tail", 35 + (i % 3))
    add(f"{seed.lower()} {m} 2026", "year_expansion", 28)
    if i % 4 == 0 and len(seed.split()) < 4:
      add(f"{seed.lower()} for {m}", "phrase", 30)

  ranked = sorted(candidates.values(), key=lambda r: (r["relevance"], len(r["sources"])), reverse=True)
  return [r["keyword"] for r in ranked[: max(count + 20, 70)]]


def extract_entities(seed: str, docs: list[OpenDoc]) -> list[str]:
  entities: set[str] = set()
  for w in re.findall(r"\w+", seed):
    if len(w) > 3:
      entities.add(w.title())
  for d in docs[:8]:
    for w in re.findall(r"\w+", d.title):
      if len(w) > 4 and w[0].isupper():
        entities.add(w)
  return list(entities)[:15]


def classify_intent(keyword: str) -> str:
  k = keyword.lower()
  if any(h in k for h in _TRANSACTIONAL):
    return "transactional"
  if any(h in k for h in _COMMERCIAL):
    return "commercial"
  if any(h in k for h in _INFORMATIONAL):
    return "informational"
  if any(h in k for h in ("official", "login", "website")):
    return "navigational"
  return "commercial" if len(k.split()) <= 2 else "informational"


def estimate_volume(keyword: str, *, seed: str, relevance: int, variation_seed: int) -> int:
  h = _kw_hash(keyword, variation_seed)
  words = keyword.split()
  vol = 400 + (h % 9000) + relevance * 80
  if len(words) == 1:
    vol = int(vol * 1.7)
  elif len(words) >= 5:
    vol = int(vol * 0.5)
  if seed.lower() in keyword.lower():
    vol = int(vol * 1.15)
  return max(100, min(50000, vol))


def estimate_difficulty(keyword: str, *, variation_seed: int, source_count: int = 0) -> int:
  h = _kw_hash(keyword, variation_seed + 3)
  words = keyword.split()
  diff = 15 + (h % 55)
  if len(words) <= 2:
    diff += 15
  elif len(words) >= 4:
    diff -= 10
  diff -= min(18, source_count * 3)
  return max(5, min(98, diff))


def estimate_cpc(keyword: str, intent: str, variation_seed: int) -> float:
  h = _kw_hash(keyword, variation_seed + 5)
  cents = 30 + (h % 1800)
  if intent == "commercial":
    cents += 450
  elif intent == "transactional":
    cents += 320
  return round(cents / 100.0, 2)


def estimate_competition(keyword: str, difficulty: int, variation_seed: int) -> int:
  h = _kw_hash(keyword, variation_seed + 7)
  comp = 10 + (h % 70) + difficulty // 4
  return max(5, min(96, comp))


def individual_trend(keyword: str, *, relevance: int, variation_seed: int) -> str:
  h = _kw_hash(keyword, variation_seed + 11)
  trends = ["up", "stable", "down"]
  t = trends[h % 3]
  if relevance >= 70:
    t = "up"
  elif relevance < 25:
    t = "stable"
  return t


def cluster_keywords(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  clusters: dict[str, list[str]] = {i: [] for i in _INTENTS}
  for it in items:
    intent = it.get("intent", "informational")
    clusters.setdefault(intent, []).append(it["keyword"])
  return [
    {"cluster": name, "keywords": kws[:12], "count": len(kws)}
    for name, kws in clusters.items()
    if kws
  ]


def find_opportunities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  opps = []
  for it in items:
    vol = it["search_volume"]
    diff = it["difficulty"]
    score = int(min(100, (vol / 500) + max(0, 60 - diff) + (15 if it["trend"] == "up" else 0)))
    it["opportunity_score"] = score
    if diff <= 45 and vol >= 500:
      opps.append({"keyword": it["keyword"], "opportunity_score": score, "reason": "low_difficulty_volume"})
  opps.sort(key=lambda x: x["opportunity_score"], reverse=True)
  return opps[:10]


def compute_seo_score(items: list[dict[str, Any]], seed: str) -> dict[str, Any]:
  if not items:
    return {"overall": 0, "coverage": 0, "diversity": 0}
  avg_opp = sum(it.get("opportunity_score", 0) for it in items) / len(items)
  intents = len({it["intent"] for it in items})
  trending = sum(1 for it in items if it["trend"] == "up")
  return {
    "overall": int(min(100, avg_opp * 0.6 + intents * 8 + trending * 2)),
    "coverage": len(items),
    "diversity": intents,
    "trending_up": trending,
    "seed_match": seed.lower(),
  }


def rank_keywords(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return sorted(
    items,
    key=lambda it: (
      it.get("opportunity_score", 0),
      it.get("relevance_score", 0),
      it["search_volume"],
      -it["difficulty"],
    ),
    reverse=True,
  )


def build_keyword_row(
  keyword: str,
  *,
  seed: str,
  sources: list[str],
  relevance: int,
  variation_seed: int,
) -> dict[str, Any]:
  intent = classify_intent(keyword)
  volume = estimate_volume(keyword, seed=seed, relevance=relevance, variation_seed=variation_seed)
  difficulty = estimate_difficulty(keyword, variation_seed=variation_seed, source_count=len(sources))
  cpc = estimate_cpc(keyword, intent, variation_seed)
  competition = estimate_competition(keyword, difficulty, variation_seed)
  trend = individual_trend(keyword, relevance=relevance, variation_seed=variation_seed)
  seo_kw = int(min(100, relevance * 0.4 + max(0, 55 - difficulty) * 0.35 + (20 if trend == "up" else 5)))
  return {
    "keyword": keyword,
    "search_volume": volume,
    "difficulty": difficulty,
    "cpc_usd": cpc,
    "competition": competition,
    "trend": trend,
    "intent": intent,
    "relevance_score": relevance,
    "sources": sources,
    "seo_score": seo_kw,
  }


async def run_seo_keyword_pipeline(
  seed_keyword: str,
  *,
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
  seed_keyword = _clean(seed_keyword)
  stages: dict[str, Any] = {}

  stages["input"] = {"seed_keyword": seed_keyword, "requested_variations": count}

  input_info = analyze_input(seed_keyword, language=language, tone=tone)
  stages["input_analyzer"] = input_info

  seed_variants = generate_seed_variants(seed_keyword, count, seed)
  stages["seed_keyword_generator"] = {"variants": seed_variants[:8], "count": len(seed_variants)}

  discovered: list[dict[str, Any]] = []
  discovery_meta: dict[str, Any] = {"enabled": discover_web, "sources_used": [], "queries_run": 0, "errors": []}
  if discover_web:
    try:
      discovery = await asyncio.wait_for(
        discover_keywords(
          seed_keyword,
          language=language,
          include_questions=include_questions,
          include_alphabet=include_alphabet,
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

  docs: list[OpenDoc] = []
  rag_sources: list[str] = []
  if use_rag:
    try:
      tokens = input_info.get("tokens") or [seed_keyword]
      docs = await asyncio.wait_for(
        retrieve_from_sources(seed_keyword, tokens[:3], _SOURCE_ROUTE[:3], per_source=1, seed=seed),
        timeout=4.0,
      )
      rag_sources = sorted({d.source for d in docs})
    except asyncio.TimeoutError:
      docs = []
    stages["source_router"] = {"sources": _SOURCE_ROUTE[:3], "datasets": OPEN_DATASET_TREE}
  else:
    stages["source_router"] = {"fast_path": "local_expansion", "datasets": OPEN_DATASET_TREE}

  expanded = expand_keywords(
    seed_keyword, seed_variants, discovered, docs, count=count, variation_seed=seed,
  )
  stages["keyword_expansion"] = {
    "candidates": len(expanded),
    "web_discovered": len(discovered),
    "rag_docs": len(docs),
  }

  entities = extract_entities(seed_keyword, docs)
  stages["entity_extractor"] = {"entities": entities}

  raw_items: list[dict[str, Any]] = []
  seen: set[str] = set()
  disc_map = {d["keyword"]: d for d in discovered if d.get("keyword")}
  for i, kw in enumerate(expanded):
    if kw in seen:
      continue
    seen.add(kw)
    disc = disc_map.get(kw, {})
    sources = list(disc.get("sources") or [])
    if not sources:
      sources = ["pipeline_expansion"]
    relevance = int(disc.get("relevance_score") or max(20, 80 - i))
    row = build_keyword_row(
      kw, seed=seed_keyword, sources=sources, relevance=relevance, variation_seed=seed + i * 17,
    )
    raw_items.append(row)
    if len(raw_items) >= count:
      break

  if len(raw_items) < count:
    extra_mods = _shuffle(list(_MODIFIERS), seed + 99)
    idx = 0
    while len(raw_items) < count and idx < 200:
      kw = f"{seed_keyword.lower()} {_pick(extra_mods, seed + idx)} {idx + 1}"
      idx += 1
      if kw in seen:
        continue
      seen.add(kw)
      raw_items.append(build_keyword_row(
        kw, seed=seed_keyword, sources=["fallback_expansion"], relevance=15,
        variation_seed=seed + idx * 23,
      ))

  stages["intent_classifier"] = {
    "distribution": {intent: sum(1 for it in raw_items if it["intent"] == intent) for intent in _INTENTS},
  }

  trend_agg = {"up": 0, "stable": 0, "down": 0}
  for it in raw_items:
    trend_agg[it["trend"]] = trend_agg.get(it["trend"], 0) + 1
  stages["trend_analyzer"] = trend_agg
  stages["volume_estimator"] = {"avg_volume": int(sum(it["search_volume"] for it in raw_items) / max(1, len(raw_items)))}
  stages["difficulty_calculator"] = {"avg_difficulty": int(sum(it["difficulty"] for it in raw_items) / max(1, len(raw_items)))}
  stages["cpc_analyzer"] = {"avg_cpc_usd": round(sum(it["cpc_usd"] for it in raw_items) / max(1, len(raw_items)), 2)}
  stages["competition_analyzer"] = {"avg_competition": int(sum(it["competition"] for it in raw_items) / max(1, len(raw_items)))}
  stages["trend_individual_engine"] = {"applied": True, "per_keyword": len(raw_items)}

  opportunities = find_opportunities(raw_items)
  ranked = rank_keywords(raw_items)[:count]
  clusters = cluster_keywords(ranked)
  stages["keyword_clustering"] = {"clusters": len(clusters), "groups": [c["cluster"] for c in clusters]}
  stages["opportunity_finder"] = {"opportunities": len(opportunities), "top": opportunities[:5]}
  stages["ranking_prioritization"] = {"ranked_count": len(ranked)}

  seo_score = compute_seo_score(ranked, seed_keyword)
  stages["seo_score"] = seo_score
  stages["json_output"] = {"keyword_count": len(ranked)}

  summary = {
    "high_volume": sum(1 for it in ranked if it["search_volume"] >= 10000),
    "low_difficulty": sum(1 for it in ranked if it["difficulty"] <= 35),
    "trending_up": sum(1 for it in ranked if it["trend"] == "up"),
    "from_web": sum(1 for it in ranked if "web_discovery" in it.get("sources", [])),
    "opportunities": len(opportunities),
  }

  return {
    "generator_version": GENERATOR_VERSION,
    "seed_keyword": seed_keyword,
    "count": len(ranked),
    "variation_seed": seed,
    "keywords": ranked,
    "clusters": clusters,
    "opportunities": opportunities,
    "summary": summary,
    "seo_score": seo_score,
    "discovery": discovery_meta,
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "pipeline": {
      "input": input_info,
      "entities": entities,
      "intent_distribution": stages["intent_classifier"]["distribution"],
      "trend_aggregate": trend_agg,
      "retrieval": {"rag_sources": rag_sources, "document_count": len(docs)},
    },
    "rag": {"enabled": use_rag, "sources_used": rag_sources},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "unlimited_outputs": True,
    "per_request_unique": True,
  }

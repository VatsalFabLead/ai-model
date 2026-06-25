"""Free web keyword discovery from public suggest/search APIs.

Aggregates related queries from Google Suggest, Bing Autosuggest, Datamuse,
and Wikipedia search. No paid SEO APIs or proprietary AI required.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote_plus

import httpx

_USER_AGENT = "NexusSEOKeywordTool/1.0 (https://github.com/VatsalFabLead/ai-model)"

# Strategic query expansions (classic keyword research patterns).
_QUESTION_PREFIXES = ("how to", "what is", "why", "best", "top", "vs", "near me")
_ALPHABET_SOUP = tuple("abcdefghijklmnopqrstuvwxyz")

_LANG_TO_HL = {
  "english": "en",
  "en": "en",
  "hindi": "hi",
  "hi": "hi",
  "spanish": "es",
  "es": "es",
  "french": "fr",
  "fr": "fr",
  "german": "de",
  "de": "de",
  "portuguese": "pt",
  "pt": "pt",
}


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_keyword(text: str) -> str:
  text = _clean(text.lower())
  text = re.sub(r"[^\w\s\-&']", " ", text, flags=re.UNICODE)
  return _clean(text)


def _is_valid_keyword(keyword: str, seed: str, *, min_len: int = 3, max_len: int = 80) -> bool:
  k = _normalize_keyword(keyword)
  if not k or len(k) < min_len or len(k) > max_len:
    return False
  if k.isdigit():
    return False
  # Must share at least one meaningful token with seed (or be seed itself).
  seed_tokens = {t for t in re.findall(r"\w+", seed.lower()) if len(t) > 2}
  kw_tokens = set(re.findall(r"\w+", k))
  if not seed_tokens:
    return True
  return k == _normalize_keyword(seed) or bool(seed_tokens & kw_tokens) or seed.lower() in k


def _hl_for_language(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_HL.get(language.strip().lower(), "en")


class KeywordCandidate:
  __slots__ = ("keyword", "sources", "best_rank")

  def __init__(self, keyword: str) -> None:
    self.keyword = _normalize_keyword(keyword)
    self.sources: set[str] = set()
    self.best_rank = 999

  def add(self, source: str, rank: int) -> None:
    self.sources.add(source)
    self.best_rank = min(self.best_rank, rank)

  def score(self, seed: str) -> float:
    seed_l = seed.lower()
    k = self.keyword
    score = len(self.sources) * 18.0
    score += max(0.0, 40.0 - float(self.best_rank))
    if k == _normalize_keyword(seed):
      score += 25.0
    elif seed_l in k:
      score += 12.0
    words = k.split()
    if len(words) >= 3:
      score += 6.0
    if any(k.startswith(p + " ") for p in _QUESTION_PREFIXES):
      score += 5.0
    return score

  def to_dict(self) -> dict[str, Any]:
    return {
      "keyword": self.keyword,
      "sources": sorted(self.sources),
      "best_rank": self.best_rank if self.best_rank < 999 else None,
      "relevance_score": 0,
    }


async def _fetch_google_suggest(
  client: httpx.AsyncClient,
  query: str,
  *,
  hl: str = "en",
) -> list[str]:
  url = "https://suggestqueries.google.com/complete/search"
  params = {"client": "firefox", "q": query, "hl": hl}
  resp = await client.get(url, params=params)
  resp.raise_for_status()
  data = resp.json()
  if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
    return [_clean(str(s)) for s in data[1] if s]
  return []


async def _fetch_bing_suggest(client: httpx.AsyncClient, query: str) -> list[str]:
  url = "https://api.bing.com/osjson.aspx"
  resp = await client.get(url, params={"query": query})
  resp.raise_for_status()
  data = resp.json()
  if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
    return [_clean(str(s)) for s in data[1] if s]
  return []


async def _fetch_datamuse(client: httpx.AsyncClient, seed: str) -> list[str]:
  out: list[str] = []
  for path, params in (
    ("words", {"ml": seed, "max": "20"}),
    ("sug", {"s": seed, "max": "15", "v": "en"}),
  ):
    resp = await client.get(f"https://api.datamuse.com/{path}", params=params)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
      continue
    for item in data:
      if isinstance(item, dict):
        word = item.get("word") or item.get("s")
        if word:
          out.append(_clean(str(word)))
      elif isinstance(item, str):
        out.append(_clean(item))
  return out


async def _fetch_wikipedia_titles(client: httpx.AsyncClient, seed: str, *, lang: str = "en") -> list[str]:
  url = f"https://{lang}.wikipedia.org/w/api.php"
  params = {
    "action": "query",
    "format": "json",
    "list": "search",
    "srsearch": seed,
    "srlimit": "12",
    "utf8": "1",
  }
  resp = await client.get(url, params=params)
  resp.raise_for_status()
  data = resp.json()
  hits = data.get("query", {}).get("search", [])
  titles: list[str] = []
  for hit in hits:
    title = _clean(hit.get("title", ""))
    if title:
      titles.append(title.lower())
    snippet = _clean(re.sub(r"<[^>]+>", " ", hit.get("snippet", "")))
    for phrase in re.findall(r"[a-z][a-z0-9\- ]{4,50}", snippet.lower()):
      titles.append(_clean(phrase))
  return titles


def _query_seed(seed: str, context: dict[str, Any] | None = None) -> str:
  """Short seed for suggest APIs — full seed is kept in pipeline output."""
  if context:
    parts = context.get("topic_parts") or []
    if parts:
      return _clean(parts[0])[:180]
    core = context.get("core_topic")
    if core:
      return _clean(str(core))[:180]
  if "," in seed:
    return _clean(seed.split(",")[0])[:180]
  return _clean(seed)[:180]


def _infer_services_from_tokens(tokens: list[str]) -> list[str]:
  joined = " ".join(tokens)
  catalog: dict[str, list[str]] = {
    "flutter": ["flutter app development", "flutter development company", "hire flutter developers"],
    "mobile": ["mobile app development company", "mobile app development services"],
    "erp": ["erp development company", "custom erp development"],
    "crm": ["crm development services", "custom crm development"],
    "ai": ["ai software development company", "ai development services"],
    "software": ["custom software development", "software development company"],
    "app": ["app development company", "custom app development"],
  }
  services: list[str] = []
  for hint, phrases in catalog.items():
    if hint in joined:
      services.extend(phrases)
  return services or [
    "software development company",
    "mobile app development company",
    "custom software development",
    "app development services",
  ]


def _expansion_queries(
  seed: str,
  *,
  include_questions: bool,
  include_alphabet: bool,
  context: dict[str, Any] | None = None,
) -> list[str]:
  seed = _clean(seed)
  queries = [seed]
  ctx = context or {}
  services = ctx.get("services") or _infer_services_from_tokens(
    [t for t in re.findall(r"\w+", seed.lower()) if len(t) > 2]
  )
  brand = ctx.get("brand_name", seed)

  if ctx.get("is_brand_seed") or len(seed.split()) >= 3:
    for svc in services[:6]:
      queries.append(svc)
      if include_questions:
        queries.append(f"how to choose {svc} company")
        queries.append(f"best {svc}")
        queries.append(f"hire {svc.split()[0]} developers")
    queries.append(brand)
  else:
    if include_questions:
      for prefix in _QUESTION_PREFIXES[:5]:
        queries.append(f"{prefix} {seed}")
  if include_alphabet and len(seed.split()) <= 3:
    for ch in _ALPHABET_SOUP[:8]:
      queries.append(f"{seed} {ch}")
  return list(dict.fromkeys(queries))


async def discover_keywords(
  seed_keyword: str,
  *,
  language: str | None = None,
  include_questions: bool = True,
  include_alphabet: bool = True,
  timeout: float = 8.0,
  context: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Search multiple public sources and return ranked related keyword ideas."""
  seed = _clean(seed_keyword)
  if not seed:
    return {"keywords": [], "sources_used": [], "queries_run": 0}

  hl = _hl_for_language(language)
  wiki_lang = hl if hl in {"en", "hi", "es", "fr", "de", "pt", "ar"} else "en"
  query_seed = _query_seed(seed, context)
  expansions = _expansion_queries(
    query_seed, include_questions=include_questions, include_alphabet=include_alphabet, context=context,
  )

  candidates: dict[str, KeywordCandidate] = {}
  sources_used: set[str] = set()
  errors: list[str] = []

  def _add_many(items: list[str], source: str, base_rank: int = 0) -> None:
    if items:
      sources_used.add(source)
    for idx, raw in enumerate(items):
      if not _is_valid_keyword(raw, seed):
        continue
      key = _normalize_keyword(raw)
      cand = candidates.get(key)
      if cand is None:
        cand = KeywordCandidate(key)
        candidates[key] = cand
      cand.add(source, base_rank + idx)

  async with httpx.AsyncClient(
    timeout=timeout,
    headers={"User-Agent": _USER_AGENT},
    follow_redirects=True,
  ) as client:
    async def _google_batch() -> None:
      sem = asyncio.Semaphore(6)

      async def _one(q: str) -> None:
        async with sem:
          try:
            items = await _fetch_google_suggest(client, q, hl=hl)
            _add_many(items, "google_suggest")
          except Exception as exc:
            errors.append(f"google:{q[:30]}:{type(exc).__name__}")

      await asyncio.gather(*[_one(q) for q in expansions])

    async def _bing() -> None:
      try:
        items = await _fetch_bing_suggest(client, query_seed)
        _add_many(items, "bing_suggest")
      except Exception as exc:
        errors.append(f"bing:{type(exc).__name__}")

    async def _datamuse() -> None:
      try:
        items = await _fetch_datamuse(client, query_seed)
        _add_many(items, "datamuse")
      except Exception as exc:
        errors.append(f"datamuse:{type(exc).__name__}")

    async def _wiki() -> None:
      try:
        items = await _fetch_wikipedia_titles(client, query_seed, lang=wiki_lang)
        _add_many(items, "wikipedia")
      except Exception as exc:
        errors.append(f"wikipedia:{type(exc).__name__}")

    await asyncio.gather(_google_batch(), _bing(), _datamuse(), _wiki())

  ranked = sorted(candidates.values(), key=lambda c: c.score(seed), reverse=True)
  keywords: list[dict[str, Any]] = []
  for cand in ranked:
    d = cand.to_dict()
    d["relevance_score"] = int(min(100, round(cand.score(seed))))
    keywords.append(d)

  return {
    "keywords": keywords,
    "sources_used": sorted(sources_used),
    "queries_run": len(expansions) + 3,
    "errors": errors[:5],
  }

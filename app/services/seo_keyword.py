"""SEO Keyword Generator — web discovery + AI expansion + ranked metrics.

For any seed keyword, searches Google/Bing suggest, Datamuse, and Wikipedia
to discover real related queries, then optionally enriches with the local model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.engine.keyword_discovery import discover_keywords
from app.services.provider_base import ModelProvider

_VALID_TONES = {
  "professional",
  "casual",
  "friendly",
  "funny",
  "excited",
  "inspirational",
  "bold",
  "informative",
  "persuasive",
  "authoritative",
  "conversational",
  "neutral",
}

_MODIFIERS = [
  "best",
  "top",
  "tools",
  "software",
  "services",
  "tips",
  "guide",
  "examples",
  "strategy",
  "tutorial",
  "checklist",
  "for beginners",
  "pricing",
  "cost",
  "vs",
  "alternatives",
  "trends 2026",
]

_LONG_TAIL_PREFIX = [
  "how to",
  "what is",
  "why",
  "when to use",
  "best way to",
]

_COMMERCIAL_HINTS = ("buy", "price", "pricing", "cost", "cheap", "deal", "shop", "hire", "agency", "service")
_TRANSACTIONAL_HINTS = ("near me", "online", "download", "free trial", "signup", "book", "order")
_INFORMATIONAL_HINTS = ("how to", "what is", "why", "guide", "tutorial", "tips", "examples", "meaning")


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "informative"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "informative"


def _seed_int(text: str) -> int:
  h = hashlib.sha256(text.encode("utf-8")).hexdigest()
  return int(h[:12], 16)


def _detect_intent(keyword: str) -> str:
  k = keyword.lower()
  if any(h in k for h in _TRANSACTIONAL_HINTS):
    return "transactional"
  if any(h in k for h in _COMMERCIAL_HINTS):
    return "commercial"
  if any(h in k for h in _INFORMATIONAL_HINTS):
    return "informational"
  return "commercial" if len(k.split()) <= 2 else "informational"


def _metrics_for_keyword(
  keyword: str,
  *,
  seed_keyword: str,
  relevance_score: int = 0,
  source_count: int = 0,
  best_rank: int | None = None,
) -> dict[str, Any]:
  s = _seed_int(keyword.lower())
  words = keyword.split()
  word_count = len(words)

  # Base estimates refined by real discovery signals when available.
  rank_boost = max(0, 35 - (best_rank or 35))
  discovery_boost = source_count * 1200 + rank_boost * 180 + relevance_score * 25

  volume = 300 + (s % 8000) + discovery_boost
  if word_count == 1:
    volume = int(volume * 1.8)
  elif word_count >= 5:
    volume = int(volume * 0.55)
  volume = max(100, min(50000, volume))

  difficulty = 12 + ((s >> 3) % 55)
  if word_count <= 2:
    difficulty += 18
  elif word_count >= 4:
    difficulty -= 12
  difficulty -= min(20, source_count * 4)
  difficulty = max(5, min(98, difficulty))

  cpc_cents = 25 + ((s >> 7) % 2200)
  if _detect_intent(keyword) == "commercial":
    cpc_cents += 400
  cpc_usd = round(cpc_cents / 100.0, 2)

  competition = 8 + ((s >> 11) % 75)
  competition = max(5, min(96, competition - source_count * 3))

  trend_map = ["up", "stable", "down"]
  trend = trend_map[(s >> 15) % 3]
  if relevance_score >= 70 and source_count >= 2:
    trend = "up"
  elif relevance_score < 25:
    trend = "stable"

  return {
    "keyword": keyword,
    "search_volume": int(volume),
    "difficulty": int(difficulty),
    "cpc_usd": cpc_usd,
    "competition": int(competition),
    "trend": trend,
    "intent": _detect_intent(keyword),
    "relevance_score": relevance_score,
    "sources": [],
  }


def _fallback_keywords(seed_keyword: str, max_items: int) -> list[str]:
  seed = _clean(seed_keyword).lower()
  out: list[str] = [seed]
  for m in _MODIFIERS:
    out.append(f"{seed} {m}")
  for p in _LONG_TAIL_PREFIX:
    out.append(f"{p} {seed}")
  uniq: list[str] = []
  seen: set[str] = set()
  for k in out:
    k = _clean(k)
    if not k or k in seen:
      continue
    seen.add(k)
    uniq.append(k)
    if len(uniq) >= max_items:
      break
  return uniq


def _parse_ai_lines(raw: str, seed_keyword: str, max_items: int) -> list[str]:
  lines = [re.sub(r"^\d+[\).\-\s]+", "", ln).strip().lower() for ln in (raw or "").splitlines()]
  cleaned: list[str] = []
  for ln in lines:
    ln = _clean(re.sub(r"^[#\-\*\•]+\s*", "", ln))
    if not ln or len(ln) < 3 or len(ln) > 80:
      continue
    cleaned.append(ln)

  if not cleaned:
    return []

  seen: set[str] = set()
  out: list[str] = []
  for k in [seed_keyword.lower(), *cleaned]:
    k = _clean(k)
    if not k or k in seen:
      continue
    seen.add(k)
    out.append(k)
    if len(out) >= max_items:
      break
  return out


def _merge_keyword_lists(
  seed_keyword: str,
  max_items: int,
  discovered: list[dict[str, Any]],
  ai_keywords: list[str],
  template_keywords: list[str],
) -> list[dict[str, Any]]:
  """Merge and rank keywords from web discovery, AI, and templates."""
  merged: dict[str, dict[str, Any]] = {}

  def _touch(
    keyword: str,
    *,
    source: str,
    relevance_score: int = 0,
    best_rank: int | None = None,
    extra_sources: list[str] | None = None,
  ) -> None:
    k = _clean(keyword.lower())
    if not k:
      return
    row = merged.get(k)
    if row is None:
      row = {
        "keyword": k,
        "sources": [],
        "relevance_score": 0,
        "best_rank": best_rank,
      }
      merged[k] = row
    if source not in row["sources"]:
      row["sources"].append(source)
    if extra_sources:
      for s in extra_sources:
        if s not in row["sources"]:
          row["sources"].append(s)
    row["relevance_score"] = max(row["relevance_score"], relevance_score)
    if best_rank is not None:
      prev = row.get("best_rank")
      row["best_rank"] = best_rank if prev is None else min(prev, best_rank)

  _touch(seed_keyword, source="seed", relevance_score=100, best_rank=0)

  for item in discovered:
    _touch(
      item["keyword"],
      source="web_discovery",
      relevance_score=int(item.get("relevance_score") or 0),
      best_rank=item.get("best_rank"),
      extra_sources=item.get("sources") or [],
    )

  for k in ai_keywords:
    _touch(k, source="ai", relevance_score=45)

  for k in template_keywords:
    _touch(k, source="template", relevance_score=20)

  ranked = sorted(
    merged.values(),
    key=lambda r: (
      r["relevance_score"],
      len(r["sources"]),
      -(r.get("best_rank") or 99),
    ),
    reverse=True,
  )

  items: list[dict[str, Any]] = []
  for row in ranked[:max_items]:
    metrics = _metrics_for_keyword(
      row["keyword"],
      seed_keyword=seed_keyword,
      relevance_score=row["relevance_score"],
      source_count=len(row["sources"]),
      best_rank=row.get("best_rank"),
    )
    metrics["sources"] = sorted(row["sources"])
    metrics["relevance_score"] = row["relevance_score"]
    items.append(metrics)
  return items


async def generate_keywords(
  provider: ModelProvider,
  *,
  seed_keyword: str,
  tone: str | None = None,
  max_items: int = 20,
  language: str | None = None,
  use_ai: bool = True,
  discover_web: bool = True,
  include_questions: bool = True,
  include_alphabet: bool = True,
) -> dict[str, Any]:
  seed_keyword = _clean(seed_keyword)
  if not seed_keyword:
    raise ValueError("seed_keyword is required")

  max_items = max(5, min(50, int(max_items)))
  tone_str = _normalize_tone(tone)
  lang_line = f" Language: {language}." if language else ""

  discovery_meta: dict[str, Any] = {
    "enabled": discover_web,
    "sources_used": [],
    "queries_run": 0,
    "errors": [],
  }
  discovered: list[dict[str, Any]] = []

  if discover_web:
    discovery = await discover_keywords(
      seed_keyword,
      language=language,
      include_questions=include_questions,
      include_alphabet=include_alphabet,
    )
    discovered = discovery.get("keywords") or []
    discovery_meta["sources_used"] = discovery.get("sources_used") or []
    discovery_meta["queries_run"] = discovery.get("queries_run") or 0
    discovery_meta["errors"] = discovery.get("errors") or []

  ai_keywords: list[str] = []
  if use_ai:
    discovered_preview = ", ".join(d["keyword"] for d in discovered[:12])
    context_line = f"\nAlready discovered from web search: {discovered_preview}" if discovered_preview else ""
    system_prompt = (
      "You are an expert SEO keyword researcher. Generate additional high-intent keyword ideas "
      "for the seed keyword. Include long-tail, commercial, and informational variants. "
      "Return plain lines only (one keyword per line), no numbering, no explanations, no markdown. "
      f"Style: {tone_str}." + lang_line
    )
    user_prompt = (
      f"Seed keyword: {seed_keyword}\n"
      f"Generate {max_items} unique keyword ideas not already in the list.{context_line}"
    )
    try:
      raw = await provider.chat(
        [{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        use_rag=False,
        skip_intent=True,
        max_tokens=min(700, 100 + max_items * 12),
        temperature=0.55,
      )
      ai_keywords = _parse_ai_lines(raw, seed_keyword, max_items)
    except Exception:
      ai_keywords = []

  template_keywords = _fallback_keywords(seed_keyword, max_items) if not discovered else []

  items = _merge_keyword_lists(
    seed_keyword,
    max_items,
    discovered,
    ai_keywords,
    template_keywords,
  )

  summary = {
    "high_volume": sum(1 for it in items if it["search_volume"] >= 10000),
    "low_difficulty": sum(1 for it in items if it["difficulty"] <= 35),
    "trending_up": sum(1 for it in items if it["trend"] == "up"),
    "from_web": sum(1 for it in items if "web_discovery" in it.get("sources", [])),
    "from_ai": sum(1 for it in items if "ai" in it.get("sources", [])),
  }

  return {
    "seed_keyword": seed_keyword,
    "count": len(items),
    "summary": summary,
    "keywords": items,
    "discovery": discovery_meta,
  }

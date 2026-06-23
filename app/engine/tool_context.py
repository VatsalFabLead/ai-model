"""Shared RAG tool context: KB, embeddings, Wikipedia, live facts."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.engine.knowledge import KnowledgeBase, load_knowledge_base, tokenize
from app.engine.knowledge import _STOPWORDS
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.engine.retriever import KnowledgeRetriever, load_retriever
from app.engine.web_knowledge import WikipediaSource


def _relevant(query: str, text: str) -> bool:
  q_words = set(tokenize(query)) - _STOPWORDS
  if not q_words:
    return True
  return not q_words.isdisjoint(set(tokenize(text)))


def _query_variants(text: str) -> list[str]:
  import re

  q = (text or "").strip()
  if not q:
    return []
  variants = [q]
  short = re.sub(
    r"^(what|who|where|when|why|how|which|tell me about|explain|define|please)\b[\s:,]*",
    "",
    q,
    flags=re.IGNORECASE,
  ).strip(" ?.!,")
  if short and short.lower() != q.lower():
    variants.append(short)
  return variants


class ToolContextBuilder:
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._kb: KnowledgeBase | None = None
    self._retriever: KnowledgeRetriever | None = None
    self._wiki: WikipediaSource | None = None
    if settings.enable_web_knowledge:
      self._wiki = WikipediaSource(
        min_words=0,
        max_words=settings.tool_context_max_words,
        timeout=settings.web_knowledge_timeout,
      )

  def load_kb(self, retriever: KnowledgeRetriever | None = None) -> None:
    self._kb = load_knowledge_base(
      knowledge_path=Path(self._settings.knowledge_path),
      corpus_path=Path(self._settings.corpus_path),
    )
    self._retriever = retriever

  async def gather(self, query: str, *, skip_wiki: bool = False) -> str:
    if not query.strip():
      return ""

    seen: set[str] = set()
    parts: list[str] = []

    live = await fetch_gold_price_context(query)
    if live:
      parts.append(live)
      seen.add(live)

    if self._kb and self._kb.size:
      for variant in _query_variants(query):
        for hit, score in self._kb.search_ranked(variant, limit=3):
          if score >= 0.15 and _relevant(query, hit) and hit not in seen:
            seen.add(hit)
            parts.append(hit)

    if self._retriever:
      answer, sim = self._retriever.retrieve(query)
      if answer and sim >= 0.35 and _relevant(query, answer) and answer not in seen:
        seen.add(answer)
        parts.append(answer)

    if not skip_wiki and self._wiki is not None and not is_gold_price_query(query):
      wiki = await self._wiki.query(query)
      if wiki and wiki not in seen:
        seen.add(wiki)
        parts.append(wiki)

    return "\n\n---\n\n".join(parts[:6])

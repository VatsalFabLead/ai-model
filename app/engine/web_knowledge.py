"""Free encyclopedia knowledge source (Wikipedia REST/Action API).

This is a FREE factual data source, NOT an AI model. It contains no GPT/Claude/
Gemini or any model — it simply reads open encyclopedia articles to provide
detailed, multilingual answers. Fully optional and toggleable via settings.
"""

from __future__ import annotations

import re

import httpx

from app.engine.knowledge import _STOPWORDS, tokenize

_USER_AGENT = "NexusCustomModel/1.0 (custom local AI; contact: self-hosted)"
# Lightweight language guess for a few common scripts/words -> Wikipedia subdomain.
_LANG_HINTS = {
  "hi": ("kya", "kaun", "kaise", "namaste", "aap", "hai", "kyun", "kahan"),
  "es": ("hola", "quien", "que", "como", "cual", "donde", "porque", "gracias"),
  "fr": ("bonjour", "qui", "quoi", "comment", "pourquoi", "quel", "merci", "ou"),
}


def guess_lang(text: str) -> str:
  low = text.lower()
  if re.search(r"[\u0900-\u097F]", text):  # Devanagari (Hindi)
    return "hi"
  if re.search(r"[\u0600-\u06FF]", text):  # Arabic
    return "ar"
  words = set(re.findall(r"\w+", low))
  for lang, hints in _LANG_HINTS.items():
    if words & set(hints):
      return lang
  return "en"


def _clean_query(text: str) -> str:
  q = text.strip()
  # Strip common question scaffolding to improve search hits.
  q = re.sub(
    r"^(what|who|where|when|why|how|which|is|are|was|were|do|does|did|tell me about|"
    r"explain|define|give me|can you tell me|please)\b[\s:,]*",
    "",
    q,
    flags=re.IGNORECASE,
  )
  q = re.sub(r"^(the|a|an|is|are|of|about)\b\s+", "", q, flags=re.IGNORECASE)
  q = q.strip(" ?.!,")
  return q or text.strip()


class WikipediaSource:
  def __init__(self, lang: str = "en", sentences: int = 8, timeout: float = 8.0) -> None:
    self._default_lang = lang
    self._sentences = max(1, min(sentences, 10))
    self._timeout = timeout
    self._cache: dict[str, str] = {}

  async def query(self, question: str) -> str | None:
    lang = guess_lang(question)
    search = _clean_query(question)
    cache_key = f"{lang}:{search.lower()}"
    if cache_key in self._cache:
      return self._cache[cache_key]

    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
      "action": "query",
      "format": "json",
      "prop": "extracts",
      "explaintext": "1",
      "exsentences": str(self._sentences),
      "redirects": "1",
      "generator": "search",
      "gsrsearch": search,
      "gsrlimit": "1",
    }
    try:
      async with httpx.AsyncClient(timeout=self._timeout) as client:
        resp = await client.get(url, params=params, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
      return None

    pages = data.get("query", {}).get("pages", {})
    extract = ""
    title = ""
    for page in pages.values():
      extract = (page.get("extract") or "").strip()
      title = (page.get("title") or "").strip()
      if extract:
        break

    if not extract or len(extract) < 40:
      return None

    # Keep just the intro (drop "== Section ==" headers and later sections).
    extract = re.split(r"\n=+\s", extract)[0].strip()
    extract = re.sub(r"\n{3,}", "\n\n", extract)

    # Relevance gate: a meaningful query word must appear in the title/extract,
    # otherwise reject (prevents irrelevant matches for gibberish queries).
    content_words = set(tokenize(question)) - _STOPWORDS
    if content_words:
      text_words = set(tokenize(f"{title} {extract}"))
      if content_words.isdisjoint(text_words):
        return None

    answer = f"**{title}**\n\n{extract}" if title else extract
    answer = f"{answer}\n\n_(Source: Wikipedia, a free encyclopedia)_"
    self._cache[cache_key] = answer
    return answer

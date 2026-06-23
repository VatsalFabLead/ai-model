"""Free encyclopedia knowledge source (Wikipedia REST/Action API).

Fetches detailed article text (no source citation in user-facing answers).
"""

from __future__ import annotations

import re

import httpx

from app.engine.answer_format import clamp_words, count_words, strip_source_attribution
from app.engine.knowledge import _STOPWORDS, tokenize

_USER_AGENT = "NexusCustomModel/1.0 (https://github.com/VatsalFabLead/ai-model)"
_LANG_HINTS = {
  "hi": ("kya", "kaun", "kaise", "namaste", "aap", "hai", "kyun", "kahan"),
  "es": ("hola", "quien", "que", "como", "cual", "donde", "porque", "gracias"),
  "fr": ("bonjour", "qui", "quoi", "comment", "pourquoi", "quel", "merci", "ou"),
}

_DISAMBIG_RE = re.compile(
  r"\b(most commonly refers to|may refer to|may also refer to)\b",
  re.IGNORECASE,
)


def guess_lang(text: str) -> str:
  low = text.lower()
  if re.search(r"[\u0900-\u097F]", text):
    return "hi"
  if re.search(r"[\u0600-\u06FF]", text):
    return "ar"
  words = set(re.findall(r"\w+", low))
  for lang, hints in _LANG_HINTS.items():
    if words & set(hints):
      return lang
  return "en"


def _strict_relevant(query: str, title: str, extract: str) -> bool:
  """Reject wrong-topic hits (e.g. Silver article for a Gold question)."""
  q_words = set(tokenize(query)) - _STOPWORDS
  text = f"{title} {extract[:800]}"
  text_words = set(tokenize(text))
  if not q_words:
    return True
  if q_words.isdisjoint(text_words):
    return False
  generic = {
    "price", "prices", "after", "years", "year", "predict", "prediction", "forecast",
    "future", "will", "can", "could", "would", "about", "tell", "give", "what",
  }
  anchors = {w for w in q_words if w not in generic and len(w) > 2}
  if anchors and anchors.isdisjoint(text_words):
    return False
  return True


def _clean_query(text: str) -> str:
  q = text.strip()
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


def _clean_extract(extract: str) -> str:
  text = (extract or "").strip()
  text = re.sub(r"\n=+\s*[^=\n]+\s*=+\n", "\n\n", text)
  text = re.sub(r"\n{3,}", "\n\n", text)
  return text.strip()


def _is_relevant(question: str, title: str, extract: str) -> bool:
  content_words = set(tokenize(question)) - _STOPWORDS
  if not content_words:
    return True
  text_words = set(tokenize(f"{title} {extract}"))
  return not content_words.isdisjoint(text_words)


class WikipediaSource:
  def __init__(
    self,
    lang: str = "en",
    *,
    min_words: int = 0,
    max_words: int = 0,
    timeout: float = 12.0,
    sentences: int | None = None,  # legacy compat; ignored when min/max words set
  ) -> None:
    self._default_lang = lang
    self._min_words = max(0, min_words)
    self._max_words = max(0, max_words)
    self._fetch_cap = self._max_words if self._max_words > 0 else 8000
    self._timeout = timeout
    self._cache: dict[str, str] = {}
    self.last_error: str | None = None
    _ = sentences

  async def query(self, question: str) -> str | None:
    lang = guess_lang(question)
    search = _clean_query(question)
    cache_key = f"{lang}:{search.lower()}:{self._min_words}:{self._max_words}"
    if cache_key in self._cache:
      return self._cache[cache_key]

    url = f"https://{lang}.wikipedia.org/w/api.php"
    # ~6 chars/word average for exchars budget per page
    chars_per_page = min(32000, self._fetch_cap * 6)
    params = {
      "action": "query",
      "format": "json",
      "prop": "extracts",
      "explaintext": "1",
      "exintro": "0",
      "exchars": str(chars_per_page),
      "redirects": "1",
      "generator": "search",
      "gsrsearch": search,
      "gsrlimit": "10",
    }
    try:
      async with httpx.AsyncClient(timeout=self._timeout) as client:
        resp = await client.get(url, params=params, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
      self.last_error = None
    except (httpx.HTTPError, ValueError) as exc:
      self.last_error = f"{type(exc).__name__}: {exc}"
      return None

    pages = data.get("query", {}).get("pages", {})
    sections: list[str] = []
    total_words = 0

    for page in sorted(pages.values(), key=lambda p: int(p.get("index", 999))):
      extract = _clean_extract(page.get("extract") or "")
      title = (page.get("title") or "").strip()
      if not extract or len(extract) < 40:
        continue
      if not _strict_relevant(question, title, extract):
        continue

      block = f"## {title}\n\n{extract}" if title else extract
      block_words = count_words(block)
      if self._max_words > 0 and total_words + block_words > self._max_words:
        remaining = self._max_words - total_words
        if remaining < 80:
          break
        block = clamp_words(block, min_words=0, max_words=remaining)
        block_words = count_words(block)

      sections.append(block)
      total_words += block_words
      if self._max_words > 0 and total_words >= self._max_words:
        break
      if self._min_words > 0 and total_words >= self._min_words and not _DISAMBIG_RE.search(extract):
        break

    if not sections:
      return None

    answer = "\n\n".join(sections)
    answer = strip_source_attribution(answer)
    if self._max_words > 0:
      answer = clamp_words(answer, min_words=0, max_words=self._max_words)
    self._cache[cache_key] = answer
    return answer

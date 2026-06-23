"""Live retrieval from free/open datasets — no GPT/Claude/Gemini.

Sources: Wikipedia, Wikidata, DBpedia, arXiv, Stack Exchange, ConceptNet,
GDELT, Semantic Scholar, GooAQ, SQuAD, Dolly (HF stream), local FAISS index.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

_USER_AGENT = "NexusSEO-RAG/1.0 (free-open-data; training-import)"


@dataclass
class OpenDoc:
  doc_id: str
  source: str
  title: str
  text: str
  url: str | None = None
  score: float = 0.0
  meta: dict[str, Any] = field(default_factory=dict)


def _clip(text: str, n: int = 1200) -> str:
  text = re.sub(r"\s+", " ", (text or "").strip())
  if len(text) <= n:
    return text
  return text[: n - 3].rstrip() + "..."


def _query_variants(topic: str, keywords: list[str], seed: int) -> list[str]:
  primary = keywords[0] if keywords else topic
  variants = [
    topic,
    primary,
    f"{topic} {primary}",
    f"{primary} guide",
    f"{primary} benefits",
    f"how to {primary}",
    f"{topic} tips",
  ]
  low = f"{topic} {primary}".lower()
  if "workout" in low or "fitness" in low or "exercise" in low:
    variants.extend(["physical exercise", "home exercise", "bodyweight exercise"])
  if "erp" in low or "enterprise resource" in low:
    variants.extend(["enterprise resource planning", "ERP system modules", "manufacturing ERP"])
  if len(keywords) > 1:
    variants.append(f"{primary} {keywords[1]}")
  # Rotate order by seed so sources get different queries
  offset = seed % len(variants)
  rotated = variants[offset:] + variants[:offset]
  seen: set[str] = set()
  out: list[str] = []
  for v in rotated:
    k = v.lower().strip()
    if k and k not in seen:
      seen.add(k)
      out.append(v.strip())
  return out[:5]


async def fetch_wikipedia(topic: str, *, limit: int = 3) -> list[OpenDoc]:
  from app.engine.web_knowledge import WikipediaSource

  wiki = WikipediaSource(min_words=0, max_words=900, timeout=12.0)
  text = await wiki.query(topic)
  if not text or len(text) < 60:
    # Fallback: plagiarism multi-chunk search
    from app.engine.plagiarism_engine import fetch_wikipedia as _wiki

    chunks = await _wiki(topic, limit=limit)
    return [
      OpenDoc(
        doc_id=f"wiki:{c.title}",
        source="wikipedia",
        title=c.title,
        text=_clip(c.text, 1200),
        url=c.url,
        score=0.85,
      )
      for c in chunks[:limit]
    ]
  title = topic.strip().title()
  return [
    OpenDoc(
      doc_id=f"wiki:{title}",
      source="wikipedia",
      title=title,
      text=_clip(text, 1500),
      url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
      score=0.9,
    )
  ]


async def fetch_wikidata(topic: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://www.wikidata.org/w/api.php",
        params={
          "action": "wbsearchentities",
          "search": topic[:80],
          "language": "en",
          "format": "json",
          "limit": limit,
        },
      )
      r.raise_for_status()
      search = r.json().get("search", [])
      ids = [item.get("id") for item in search if item.get("id")]
      if not ids:
        return []
      r2 = await client.get(
        "https://www.wikidata.org/w/api.php",
        params={
          "action": "wbgetentities",
          "ids": "|".join(ids),
          "props": "descriptions|labels",
          "languages": "en",
          "format": "json",
        },
      )
      r2.raise_for_status()
      entities = r2.json().get("entities", {})
    for eid, ent in entities.items():
      label = (ent.get("labels", {}).get("en", {}) or {}).get("value", "")
      desc = (ent.get("descriptions", {}).get("en", {}) or {}).get("value", "")
      if label and desc:
        docs.append(
          OpenDoc(
            doc_id=f"wikidata:{eid}",
            source="wikidata",
            title=label,
            text=_clip(f"{label}: {desc}", 800),
            url=f"https://www.wikidata.org/wiki/{eid}",
            score=0.75,
          )
        )
  except Exception:
    pass
  return docs


async def fetch_dbpedia(topic: str, *, limit: int = 3) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://lookup.dbpedia.org/api/search",
        params={"query": topic[:80], "format": "json", "maxResults": limit},
      )
      if r.status_code != 200:
        return []
      results = r.json() if r.content else []
      if isinstance(results, dict):
        results = results.get("results", results.get("docs", []))
      for item in (results or [])[:limit]:
        label = (item.get("label") or item.get("resource") or "").strip()
        comment = (item.get("comment") or item.get("description") or "").strip()
        resource = item.get("resource", "")
        if label and comment:
          docs.append(
            OpenDoc(
              doc_id=f"dbpedia:{label[:30]}",
              source="dbpedia",
              title=label,
              text=_clip(comment, 1500),
              url=resource if resource.startswith("http") else f"https://dbpedia.org/page/{label.replace(' ', '_')}",
              score=0.78,
            )
          )
  except Exception:
    pass
  return docs[:limit]


async def fetch_arxiv(topic: str, *, limit: int = 3) -> list[OpenDoc]:
  import xml.etree.ElementTree as ET

  params = {
    "search_query": f"all:{topic[:80]}",
    "start": 0,
    "max_results": limit,
    "sortBy": "relevance",
  }
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=25.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get("http://export.arxiv.org/api/query", params=params)
      r.raise_for_status()
      root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
      title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
      summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
      link = entry.find("atom:id", ns)
      url = link.text if link is not None else None
      if title and summary:
        docs.append(
          OpenDoc(
            doc_id=f"arxiv:{title[:40]}",
            source="arxiv",
            title=title,
            text=_clip(summary, 1200),
            url=url,
            score=0.72,
          )
        )
  except Exception:
    pass
  return docs


async def fetch_stackexchange(topic: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://api.stackexchange.com/2.3/search/advanced",
        params={
          "order": "desc",
          "sort": "relevance",
          "q": topic[:100],
          "site": "stackoverflow",
          "pagesize": limit,
          "filter": "withbody",
        },
      )
      r.raise_for_status()
      data = r.json()
    for item in data.get("items", []):
      title = (item.get("title") or "").strip()
      body = re.sub(r"<[^>]+>", " ", item.get("body") or "")
      body = _clip(body, 900)
      qid = item.get("question_id")
      if title and body:
        docs.append(
          OpenDoc(
            doc_id=f"stackexchange:{qid}",
            source="stackexchange",
            title=title,
            text=body,
            url=item.get("link"),
            score=0.7,
          )
        )
  except Exception:
    pass
  return docs


async def fetch_conceptnet(topic: str, *, limit: int = 5) -> list[OpenDoc]:
  term = re.sub(r"\s+", "_", topic.lower().strip())[:40]
  url = f"http://api.conceptnet.io/c/en/{term}"
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(url)
      if r.status_code != 200:
        return []
      data = r.json()
    edges = data.get("edges", [])[:limit * 2]
    lines: list[str] = []
    for e in edges:
      start = e.get("start", {}).get("label", "")
      rel = e.get("rel", {}).get("label", "").replace("/", " ")
      end = e.get("end", {}).get("label", "")
      if start and end:
        lines.append(f"{start} {rel} {end}")
    if lines:
      docs.append(
        OpenDoc(
          doc_id=f"conceptnet:{term}",
          source="conceptnet",
          title=f"ConceptNet: {topic}",
          text=_clip(". ".join(lines), 1000),
          url=f"https://conceptnet.io/c/en/{term}",
          score=0.65,
        )
      )
  except Exception:
    pass
  return docs[:limit]


async def fetch_gdelt(topic: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
          "query": topic[:80],
          "mode": "ArtList",
          "maxrecords": str(limit),
          "format": "json",
        },
      )
      r.raise_for_status()
      data = r.json()
    for art in data.get("articles", [])[:limit]:
      title = (art.get("title") or "").strip()
      url = art.get("url")
      seendate = art.get("seendate", "")
      if title:
        docs.append(
          OpenDoc(
            doc_id=f"gdelt:{hash(title) & 0xFFFFFF}",
            source="gdelt",
            title=title,
            text=_clip(f"{title}. Published {seendate}.", 400),
            url=url,
            score=0.68,
          )
        )
  except Exception:
    pass
  return docs


async def fetch_semantic_scholar(topic: str, *, limit: int = 3) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
      r = await client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": topic[:80], "limit": limit, "fields": "title,abstract,url"},
      )
      if r.status_code != 200:
        return []
      data = r.json()
    for paper in data.get("data", []):
      title = (paper.get("title") or "").strip()
      abstract = (paper.get("abstract") or "").strip()
      if title and abstract:
        docs.append(
          OpenDoc(
            doc_id=f"s2:{paper.get('paperId', title[:20])}",
            source="semantic_scholar",
            title=title,
            text=_clip(abstract, 1200),
            url=paper.get("url"),
            score=0.74,
          )
        )
  except Exception:
    pass
  return docs


async def fetch_hf_keyword(
  dataset: str,
  config: str | None,
  topic: str,
  *,
  text_field: str,
  q_field: str | None = None,
  a_field: str | None = None,
  max_scan: int = 80,
  limit: int = 3,
) -> list[OpenDoc]:
  """Stream-scan HF dataset for keyword matches (free tiers only)."""
  docs: list[OpenDoc] = []
  try:
    from datasets import load_dataset  # type: ignore
  except ImportError:
    return []

  terms = {t.lower() for t in re.findall(r"\w+", topic) if len(t) > 3}
  if not terms:
    return []

  def _match(text: str) -> bool:
    low = text.lower()
    return sum(1 for t in terms if t in low) >= min(2, len(terms))

  try:
    kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if config:
      ds = load_dataset(dataset, config, **kwargs)
    else:
      ds = load_dataset(dataset, **kwargs)
    scanned = 0
    for row in ds:
      scanned += 1
      if scanned > max_scan:
        break
      if q_field and a_field:
        q = str(row.get(q_field) or "")
        a = str(row.get(a_field) or "")
        if not _match(q + " " + a):
          continue
        docs.append(
          OpenDoc(
            doc_id=f"{dataset}:{scanned}",
            source=dataset.split("/")[-1],
            title=_clip(q, 120),
            text=_clip(a, 1200),
            score=0.66,
          )
        )
      else:
        text = str(row.get(text_field) or "")
        if len(text) < 80 or not _match(text):
          continue
        docs.append(
          OpenDoc(
            doc_id=f"{dataset}:{scanned}",
            source=dataset.split("/")[-1],
            title=_clip(text[:80], 80),
            text=_clip(text, 1200),
            score=0.64,
          )
        )
      if len(docs) >= limit:
        break
  except Exception:
    pass
  return docs


async def fetch_gooaq(topic: str, *, limit: int = 2) -> list[OpenDoc]:
  return await fetch_hf_keyword(
    "sentence-transformers/gooaq", None, topic,
    text_field="answer", q_field="question", a_field="answer", limit=limit,
  )


async def fetch_squad(topic: str, *, limit: int = 2) -> list[OpenDoc]:
  return await fetch_hf_keyword(
    "rajpurkar/squad", "plain_text", topic,
    text_field="context", q_field="question", a_field="context", limit=limit, max_scan=300,
  )


async def fetch_dolly(topic: str, *, limit: int = 2) -> list[OpenDoc]:
  return await fetch_hf_keyword(
    "databricks/databricks-dolly-15k", None, topic,
    text_field="response", q_field="instruction", a_field="response", limit=limit, max_scan=350,
  )


async def fetch_c4_snippet(topic: str, *, limit: int = 2) -> list[OpenDoc]:
  return await fetch_hf_keyword(
    "allenai/c4", "en", topic, text_field="text", limit=limit, max_scan=250,
  )


async def fetch_fineweb_snippet(topic: str, *, limit: int = 2) -> list[OpenDoc]:
  return await fetch_hf_keyword(
    "HuggingFaceFW/fineweb", "sample-10BT", topic, text_field="text", limit=limit, max_scan=200,
  )


async def fetch_local_faiss(topic: str, *, limit: int = 4) -> list[OpenDoc]:
  docs: list[OpenDoc] = []
  try:
    from app.engine.plagiarism_engine import embed_texts, search_faiss

    vecs = embed_texts([topic])
    if vecs is None or len(vecs) == 0:
      return []
    for score, meta in search_faiss(vecs[0], top_k=limit):
      text = meta.get("text", "")
      if len(text) < 40:
        continue
      docs.append(
        OpenDoc(
          doc_id=f"faiss:{meta.get('title', '')[:30]}",
          source=meta.get("source", "local_index"),
          title=meta.get("title", "Indexed document"),
          text=_clip(text, 1200),
          url=meta.get("url"),
          score=float(score) * 0.9,
        )
      )
  except Exception:
    pass
  return docs


SOURCE_FETCHERS: dict[str, Any] = {
  "wikipedia": fetch_wikipedia,
  "wikidata": fetch_wikidata,
  "dbpedia": fetch_dbpedia,
  "arxiv": fetch_arxiv,
  "stackexchange": fetch_stackexchange,
  "conceptnet": fetch_conceptnet,
  "gdelt": fetch_gdelt,
  "semantic_scholar": fetch_semantic_scholar,
  "gooaq": fetch_gooaq,
  "squad": fetch_squad,
  "dolly": fetch_dolly,
  "c4": fetch_c4_snippet,
  "fineweb": fetch_fineweb_snippet,
  "local_faiss": fetch_local_faiss,
}

# Topic class → ordered source list (free only; fast APIs first)
SOURCE_ROUTES: dict[str, list[str]] = {
  "general": ["wikipedia", "wikidata", "dbpedia", "local_faiss", "gooaq", "squad"],
  "technical": ["wikipedia", "arxiv", "semantic_scholar", "stackexchange", "local_faiss"],
  "news": ["wikipedia", "gdelt", "c4", "fineweb"],
  "health_fitness": ["wikipedia", "wikidata", "dbpedia", "conceptnet", "local_faiss", "gooaq"],
  "programming": ["stackexchange", "wikipedia", "arxiv", "local_faiss"],
  "how_to": ["wikipedia", "wikidata", "conceptnet", "dolly", "gooaq"],
  "business": ["wikipedia", "gdelt", "local_faiss", "gooaq"],
  "enterprise": ["wikipedia", "wikidata", "dbpedia", "stackexchange", "semantic_scholar", "local_faiss"],
}

_HF_SOURCES = frozenset({"gooaq", "squad", "dolly", "c4", "fineweb"})
_SOURCE_TIMEOUT_SEC = 10.0
_HF_TIMEOUT_SEC = 6.0


async def retrieve_from_sources(
  topic: str,
  keywords: list[str],
  sources: list[str],
  *,
  per_source: int = 2,
  seed: int = 0,
) -> list[OpenDoc]:
  queries = _query_variants(topic, keywords, seed)

  async def _one(src: str, q: str) -> list[OpenDoc]:
    fn = SOURCE_FETCHERS.get(src)
    if not fn:
      return []
    timeout = _HF_TIMEOUT_SEC if src in _HF_SOURCES else _SOURCE_TIMEOUT_SEC
    try:
      return await asyncio.wait_for(fn(q, limit=per_source), timeout=timeout)
    except Exception:
      return []

  fast = [s for s in sources if s not in _HF_SOURCES]
  slow = [s for s in sources if s in _HF_SOURCES]
  docs: list[OpenDoc] = []

  async def _batch(srcs: list[str]) -> list[OpenDoc]:
    out: list[OpenDoc] = []
    tasks = []
    for src in srcs:
      idx = (seed + hash(src)) % len(queries)
      tasks.append(_one(src, queries[idx]))
    for batch in await asyncio.gather(*tasks, return_exceptions=True):
      if isinstance(batch, list):
        out.extend(batch)
    return out

  docs.extend(await _batch(fast))
  # Skip slow HF streams when fast APIs already returned enough evidence
  if len(docs) < 3 and slow:
    docs.extend(await _batch(slow))
  return docs

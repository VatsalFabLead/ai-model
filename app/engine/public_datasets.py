"""Public dataset registry and samplers for custom Nexus training.

Only sources marked as free (or free-with-attribution) are enabled by default.
No GPT, Claude, or Gemini — raw text only, converted to KB + corpus chunks.

Tiers:
  fully_free        — CC0 / public domain / arXiv license
  free_attribution  — CC BY-SA, ODC-By, etc. (attribution recorded in data/ATTRIBUTIONS.md)
  gated             — HF gated; permissive licenses only (e.g. StarCoderData)
  mixed             — composite dumps with heterogeneous licenses (opt-in only)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Iterator

import httpx

_USER_AGENT = "NexusCustomModel/1.0 (https://github.com/VatsalFabLead/ai-model; training-import)"


@dataclass(frozen=True)
class DatasetSpec:
  id: str
  name: str
  tier: str  # fully_free | free_attribution | gated | mixed
  license: str
  attribution: str
  notes: str


DATASETS: dict[str, DatasetSpec] = {
  "wikidata": DatasetSpec(
    id="wikidata",
    name="Wikidata",
    tier="fully_free",
    license="CC0 1.0",
    attribution="Wikidata contributors (CC0)",
    notes="Structured facts via Wikidata Query Service.",
  ),
  "gutenberg": DatasetSpec(
    id="gutenberg",
    name="Project Gutenberg",
    tier="fully_free",
    license="Public Domain (US)",
    attribution="Project Gutenberg",
    notes="Classic public-domain books.",
  ),
  "arxiv": DatasetSpec(
    id="arxiv",
    name="arXiv",
    tier="fully_free",
    license="arXiv.org perpetual non-exclusive license",
    attribution="arXiv.org",
    notes="Research paper titles and abstracts.",
  ),
  "stackexchange": DatasetSpec(
    id="stackexchange",
    name="Stack Exchange",
    tier="free_attribution",
    license="CC BY-SA 4.0",
    attribution="Stack Exchange Inc. (CC BY-SA 4.0)",
    notes="Q&A via the public Stack Exchange API.",
  ),
  "fineweb": DatasetSpec(
    id="fineweb",
    name="FineWeb",
    tier="free_attribution",
    license="ODC-By 1.0 (+ Common Crawl Terms of Use)",
    attribution="HuggingFaceFW/fineweb (ODC-By 1.0)",
    notes="Filtered web text; requires `pip install datasets` and streams a small sample.",
  ),
  "starcoder": DatasetSpec(
    id="starcoder",
    name="StarCoderData (GitHub)",
    tier="gated",
    license="Per-repo permissive licenses + The Stack Terms of Use",
    attribution="BigCode / The Stack (per-file licenses)",
    notes="Gated on Hugging Face; only permissively licensed code is included.",
  ),
  "pile": DatasetSpec(
    id="pile",
    name="The Pile",
    tier="mixed",
    license="Mixed (per-component licenses)",
    attribution="EleutherAI / component sources",
    notes="Composite dataset; some subsets have restrictions. Opt-in only.",
  ),
  "redpajama": DatasetSpec(
    id="redpajama",
    name="RedPajama",
    tier="mixed",
    license="Mixed (per-component licenses)",
    attribution="Together AI / component sources",
    notes="Composite web dump; opt-in only.",
  ),
  "gooaq": DatasetSpec(
    id="gooaq",
    name="GooAQ",
    tier="fully_free",
    license="CC BY-SA 4.0",
    attribution="GooAQ (Google)",
    notes="Open Q&A from Google search snippets.",
  ),
  "c4": DatasetSpec(
    id="c4",
    name="C4 (Colossal Clean Crawled Corpus)",
    tier="free_attribution",
    license="ODC-By / Common Crawl Terms",
    attribution="AllenAI / Common Crawl",
    notes="Cleaned Common Crawl text; HF streaming sample.",
  ),
  "openwebtext": DatasetSpec(
    id="openwebtext",
    name="OpenWebText",
    tier="free_attribution",
    license="Research use / web-derived",
    attribution="OpenWebText corpus",
    notes="Reddit-outbound web text sample.",
  ),
  "datacommons": DatasetSpec(
    id="datacommons",
    name="Data Commons",
    tier="fully_free",
    license="CC BY / open data terms",
    attribution="Data Commons (Google)",
    notes="Public statistics via Data Commons API.",
  ),
  "commoncrawl": DatasetSpec(
    id="commoncrawl",
    name="Common Crawl",
    tier="free_attribution",
    license="Common Crawl Terms of Use",
    attribution="Common Crawl Foundation",
    notes="Use via C4/FineWeb samples (full WARC is huge).",
  ),
  "laion": DatasetSpec(
    id="laion",
    name="LAION-5B",
    tier="mixed",
    license="Creative Commons (per-image)",
    attribution="LAION",
    notes="Image-text pairs; opt-in for multimodal only.",
  ),
  "kaggle": DatasetSpec(
    id="kaggle",
    name="Kaggle Datasets",
    tier="mixed",
    license="Per-dataset on Kaggle",
    attribution="Kaggle / dataset authors",
    notes="Requires Kaggle API + per-dataset license check; opt-in only.",
  ),
}


def allowed_tiers(*, only_free: bool, include_gated: bool, include_mixed: bool) -> set[str]:
  tiers = {"fully_free"}
  if only_free:
    tiers.add("free_attribution")
  if include_gated:
    tiers.add("gated")
  if include_mixed:
    tiers.add("mixed")
  return tiers


def list_datasets(
  *,
  only_free: bool = True,
  include_gated: bool = False,
  include_mixed: bool = False,
) -> list[DatasetSpec]:
  tiers = allowed_tiers(
    only_free=only_free,
    include_gated=include_gated,
    include_mixed=include_mixed,
  )
  return [d for d in DATASETS.values() if d.tier in tiers]


def _clip(text: str, max_chars: int) -> str:
  text = re.sub(r"\s+", " ", (text or "").strip())
  if len(text) <= max_chars:
    return text
  return text[: max_chars - 3].rstrip() + "..."


def _http(timeout: float = 30.0) -> httpx.Client:
  return httpx.Client(
    timeout=timeout,
    headers={"User-Agent": _USER_AGENT},
    follow_redirects=True,
  )


def sample_wikidata(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  query = f"""
SELECT ?itemLabel ?itemDescription WHERE {{
  ?item wdt:P31 wd:Q5 .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {max(1, min(limit, 50))}
""".strip()
  url = "https://query.wikidata.org/sparql"
  with _http(timeout=60.0) as client:
    r = client.get(url, params={"query": query, "format": "json"})
    r.raise_for_status()
    data = r.json()
  for row in data.get("results", {}).get("bindings", []):
    label = row.get("itemLabel", {}).get("value", "").strip()
    desc = row.get("itemDescription", {}).get("value", "").strip()
    if not label or label.endswith("Q") or not desc:
      continue
    yield {
      "q": f"Who is {label}?",
      "a": _clip(f"**{label}** — {desc}", max_chars),
      "source": "wikidata",
    }


GUTENBERG_IDS = (11, 84, 1342, 1661, 2701, 74, 98, 1952)


def sample_gutenberg(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  count = 0
  with _http() as client:
    for book_id in GUTENBERG_IDS:
      if count >= limit:
        break
      url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
      try:
        r = client.get(url)
        r.raise_for_status()
      except httpx.HTTPError:
        continue
      raw = r.text
      title_m = re.search(r"Title:\s*(.+)", raw)
      author_m = re.search(r"Author:\s*(.+)", raw)
      title = title_m.group(1).strip() if title_m else f"Book {book_id}"
      author = author_m.group(1).strip() if author_m else "Unknown"
      body = raw.split("*** START OF", 1)[-1]
      body = body.split("*** END OF", 1)[0]
      body = re.sub(r"\s+", " ", body).strip()
      excerpt = _clip(body, max_chars)
      yield {
        "q": f"Summary of {title}",
        "a": _clip(f"**{title}** by {author}.\n\n{excerpt}", max_chars + 200),
        "source": "gutenberg",
      }
      count += 1


def sample_arxiv(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  url = "http://export.arxiv.org/api/query"
  params = {
    "search_query": "cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
    "start": 0,
    "max_results": max(1, min(limit, 50)),
    "sortBy": "submittedDate",
    "sortOrder": "descending",
  }
  ns = {"atom": "http://www.w3.org/2005/Atom"}
  with _http() as client:
    r = client.get(url, params=params)
    r.raise_for_status()
    root = ET.fromstring(r.text)
  for entry in root.findall("atom:entry", ns):
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    if not title or not summary:
      continue
    yield {
      "q": f"What is the paper '{title}' about?",
      "a": _clip(f"**{title}**\n\n{summary}", max_chars),
      "source": "arxiv",
    }


def sample_stackexchange(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  url = "https://api.stackexchange.com/2.3/questions"
  params = {
    "order": "desc",
    "sort": "votes",
    "site": "stackoverflow",
    "pagesize": max(1, min(limit, 50)),
    "filter": "withbody",
  }
  with _http() as client:
    r = client.get(url, params=params)
    r.raise_for_status()
    data = r.json()
  for item in data.get("items", []):
    title = (item.get("title") or "").strip()
    body = re.sub(r"<[^>]+>", " ", item.get("body") or "")
    body = _clip(body, max_chars // 2)
    if not title:
      continue
    yield {
      "q": title,
      "a": _clip(body or "See Stack Overflow for community answers.", max_chars),
      "source": "stackexchange",
    }


def _require_datasets():
  try:
    from datasets import load_dataset  # type: ignore
  except ImportError as exc:
    raise RuntimeError(
      "Install Hugging Face datasets for this source: pip install datasets"
    ) from exc
  return load_dataset


def sample_fineweb(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
  count = 0
  for row in ds:
    text = _clip(str(row.get("text") or ""), max_chars)
    if len(text) < 80:
      continue
    yield {
      "q": f"Explain this topic: {_clip(text[:80], 80)}",
      "a": text,
      "source": "fineweb",
    }
    count += 1
    if count >= limit:
      break


def sample_starcoder(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
  count = 0
  for row in ds:
    content = _clip(str(row.get("content") or ""), max_chars)
    if len(content) < 40:
      continue
    path = str(row.get("max_stars_repo_path") or row.get("path") or "python/code")
    yield {
      "q": f"Explain this {path} code pattern",
      "a": f"```python\n{content}\n```",
      "source": "starcoder",
    }
    count += 1
    if count >= limit:
      break


def sample_pile(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
  count = 0
  for row in ds:
    text = _clip(str(row.get("text") or ""), max_chars)
    if len(text) < 80:
      continue
    meta = str(row.get("meta") or row.get("pile_set_name") or "pile")
    yield {
      "q": f"Summarize this {meta} excerpt",
      "a": text,
      "source": "pile",
    }
    count += 1
    if count >= limit:
      break


def sample_redpajama(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset(
    "togethercomputer/RedPajama-Data-V2",
    name="default",
    split="train",
    streaming=True,
  )
  count = 0
  for row in ds:
    text = _clip(str(row.get("raw_content") or row.get("text") or ""), max_chars)
    if len(text) < 80:
      continue
    yield {
      "q": f"Summarize this web excerpt: {_clip(text[:60], 60)}",
      "a": text,
      "source": "redpajama",
    }
    count += 1
    if count >= limit:
      break


def sample_gooaq(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("sentence-transformers/gooaq", split="train", streaming=True)
  count = 0
  for row in ds:
    q = _clip(str(row.get("question") or ""), 200)
    a = _clip(str(row.get("answer") or ""), max_chars)
    if len(q) < 8 or len(a) < 40:
      continue
    yield {"q": q, "a": a, "source": "gooaq"}
    count += 1
    if count >= limit:
      break


def sample_c4(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
  count = 0
  for row in ds:
    text = _clip(str(row.get("text") or ""), max_chars)
    if len(text) < 80:
      continue
    yield {
      "q": f"Summarize: {_clip(text[:70], 70)}",
      "a": text,
      "source": "c4",
    }
    count += 1
    if count >= limit:
      break


def sample_openwebtext(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  load_dataset = _require_datasets()
  ds = load_dataset("openwebtext", split="train", streaming=True)
  count = 0
  for row in ds:
    text = _clip(str(row.get("text") or ""), max_chars)
    if len(text) < 80:
      continue
    yield {"q": f"Explain this topic: {_clip(text[:60], 60)}", "a": text, "source": "openwebtext"}
    count += 1
    if count >= limit:
      break


def sample_datacommons(limit: int, max_chars: int) -> Iterator[dict[str, str]]:
  """Sample public statistics statements from Data Commons."""
  queries = [
    "India population",
    "United States GDP",
    "world population",
    "gold production",
    "life expectancy India",
  ]
  url = "https://api.datacommons.org/v2/observation"
  count = 0
  with _http(timeout=20.0) as client:
    for topic in queries:
      if count >= limit:
        break
      try:
        r = client.get(
          url,
          params={
            "date": "LATEST",
            "variable": "Count_Person",
            "entity": "country/IND" if "India" in topic else "Earth",
          },
        )
        if r.status_code != 200:
          continue
        data = r.json()
        blob = _clip(str(data)[:max_chars], max_chars)
        if len(blob) < 40:
          continue
        yield {"q": f"What does Data Commons say about {topic}?", "a": blob, "source": "datacommons"}
        count += 1
      except httpx.HTTPError:
        continue


SAMPLERS: dict[str, Callable[[int, int], Iterator[dict[str, str]]]] = {
  "wikidata": sample_wikidata,
  "gutenberg": sample_gutenberg,
  "arxiv": sample_arxiv,
  "stackexchange": sample_stackexchange,
  "fineweb": sample_fineweb,
  "starcoder": sample_starcoder,
  "pile": sample_pile,
  "redpajama": sample_redpajama,
  "gooaq": sample_gooaq,
  "c4": sample_c4,
  "openwebtext": sample_openwebtext,
  "datacommons": sample_datacommons,
  "commoncrawl": sample_c4,
}


def sample_dataset(
  dataset_id: str,
  *,
  limit: int = 20,
  max_chars: int = 1200,
) -> list[dict[str, str]]:
  fn = SAMPLERS.get(dataset_id)
  if not fn:
    raise KeyError(f"Unknown dataset: {dataset_id}")
  return list(fn(limit, max_chars))

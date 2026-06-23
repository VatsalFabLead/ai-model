"""Plagiarism pipeline — Sentence Transformers + FAISS + free live sources.

Workflow:
  User Text → Preprocessing → Sentence Splitting → Embeddings
  → FAISS index + Wikipedia / blogs (SearXNG) / PDFs & corpus
  → Similarity Engine → Match Ranking → Highlighted Sentences → Report
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "plagiarism"
META_FILE = "meta.jsonl"
INDEX_FILE = "index.faiss"

_USER_AGENT = "NexusPlagiarismChecker/1.0 (https://github.com/VatsalFabLead/ai-model)"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MATCH_THRESHOLD = 0.65
_HIGH_MATCH = 0.78

_model = None
_index = None
_meta: list[dict[str, str]] = []
_index_dir: Path = DEFAULT_INDEX_DIR


@dataclass
class CorpusChunk:
  text: str
  source: str
  title: str
  url: str


@dataclass
class SentenceMatch:
  sentence: str
  sentence_index: int
  score: float
  match_percent: int
  source: str
  source_type: str
  title: str
  url: str
  matched_excerpt: str
  label: str


@dataclass
class PipelineResult:
  workflow: list[dict[str, str]]
  preprocessed_text: str
  sentences: list[str]
  highlighted_sentences: list[dict[str, Any]]
  matched_segments: list[dict[str, Any]]
  similarity_percent: int
  original_percent: int
  sources_used: list[str]
  avg_embedding_similarity: float
  chunks_scanned: int
  chunks_matched: int
  available: bool
  scan_incomplete: bool = False
  error: str = ""


def configure(index_dir: Path | None = None) -> None:
  global _index_dir
  if index_dir:
    _index_dir = index_dir


def _require_st():
  try:
    from sentence_transformers import SentenceTransformer
  except ImportError as exc:
    raise ImportError(
      "Install plagiarism stack: pip install sentence-transformers faiss-cpu"
    ) from exc
  return SentenceTransformer


def _require_faiss():
  try:
    import faiss
  except ImportError as exc:
    raise ImportError(
      "Install plagiarism stack: pip install sentence-transformers faiss-cpu"
    ) from exc
  return faiss


def is_available() -> bool:
  try:
    _require_st()
    _require_faiss()
    return (_index_dir / INDEX_FILE).exists() and (_index_dir / META_FILE).exists()
  except ImportError:
    return False


# ── 1. Text preprocessing ────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
  """Normalize whitespace, strip markup, unify unicode."""
  t = unicodedata.normalize("NFKC", text or "")
  t = re.sub(r"<[^>]+>", " ", t)
  t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)  # markdown links
  t = re.sub(r"[`*_#]+", "", t)
  t = re.sub(r"\s+", " ", t).strip()
  return t


# ── 2. Sentence splitting ────────────────────────────────────────────────────

def split_sentences(text: str, *, min_len: int = 12) -> list[str]:
  parts = re.split(r"(?<=[.!?])\s+", text.strip())
  return [s.strip() for s in parts if len(s.strip()) >= min_len]


def chunk_sentences(sentences: list[str], *, max_len: int = 280) -> list[str]:
  """Merge or split for embedding windows."""
  if not sentences:
    return []
  out: list[str] = []
  for s in sentences:
    if len(s) <= max_len:
      out.append(s)
    else:
      words = s.split()
      buf: list[str] = []
      for w in words:
        if sum(len(x) + 1 for x in buf) + len(w) > max_len and buf:
          out.append(" ".join(buf))
          buf = [w]
        else:
          buf.append(w)
      if buf:
        out.append(" ".join(buf))
  return out


# ── 3. Embedding generator ───────────────────────────────────────────────────

def warm_up() -> bool:
  """Preload embedding model at server startup (avoids first-request timeout)."""
  try:
    _get_model()
    return True
  except Exception:
    return False


def _get_model():
  global _model
  if _model is None:
    SentenceTransformer = _require_st()
    _model = SentenceTransformer(_MODEL_NAME)
  return _model


def generate_embeddings(texts: list[str]) -> np.ndarray:
  if not texts:
    return np.zeros((0, 384), dtype=np.float32)
  model = _get_model()
  vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
  return np.asarray(vecs, dtype=np.float32)


embed_texts = generate_embeddings  # backward compat


# ── 4. Vector database (FAISS) ───────────────────────────────────────────────

def _load_index() -> bool:
  global _index, _meta
  if _index is not None:
    return True
  meta_path = _index_dir / META_FILE
  index_path = _index_dir / INDEX_FILE
  if not meta_path.exists() or not index_path.exists():
    return False
  faiss = _require_faiss()
  _index = faiss.read_index(str(index_path))
  _meta = []
  with meta_path.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        _meta.append(json.loads(line))
  return _index.ntotal > 0


def search_faiss(query_vec: np.ndarray, top_k: int = 5) -> list[tuple[float, dict[str, str]]]:
  if not _load_index() or _index is None:
    return []
  q = query_vec.reshape(1, -1).astype(np.float32)
  scores, ids = _index.search(q, min(top_k, _index.ntotal))
  hits: list[tuple[float, dict[str, str]]] = []
  for score, idx in zip(scores[0], ids[0]):
    if idx < 0 or idx >= len(_meta):
      continue
    hits.append((float(score), _meta[idx]))
  return hits


def build_index(chunks: list[CorpusChunk], *, index_dir: Path | None = None) -> int:
  """Build FAISS index from corpus chunks (Wikipedia, PDFs, blogs, corpus)."""
  global _index, _meta, _model
  target = index_dir or _index_dir
  target.mkdir(parents=True, exist_ok=True)

  if not chunks:
    return 0

  faiss = _require_faiss()
  texts = [c.text for c in chunks]
  vecs = generate_embeddings(texts)
  dim = vecs.shape[1]

  index = faiss.IndexFlatIP(dim)
  index.add(vecs)

  meta_path = target / META_FILE
  with meta_path.open("w", encoding="utf-8") as f:
    for c in chunks:
      f.write(json.dumps({
        "text": c.text,
        "source": c.source,
        "title": c.title,
        "url": c.url,
      }, ensure_ascii=False) + "\n")

  faiss.write_index(index, str(target / INDEX_FILE))
  _index = index
  _meta = [{"text": c.text, "source": c.source, "title": c.title, "url": c.url} for c in chunks]
  return len(chunks)


# ── 5. External sources: Wikipedia + blogs (SearXNG) ─────────────────────────

def _query_from_text(text: str) -> str:
  words = re.findall(r"\w+", text.lower())
  stop = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has"}
  key = [w for w in words if w not in stop and len(w) > 3][:12]
  return " ".join(key) if key else text[:80]


def _topic_queries(text: str) -> list[str]:
  """Leading topic phrases — e.g. 'Artificial intelligence' from the first sentence."""
  queries: list[str] = []
  first = (text or "").strip().split(".")[0].strip()
  m = re.match(r"^([A-Z][^.!?()]{4,90}?)(?:\s*\([^)]+\))?", first)
  if m:
    queries.append(m.group(1).strip())
  m2 = re.match(r"^([A-Z][a-z]+(?:\s+[a-z]+){0,4})", first)
  if m2 and m2.group(1) not in queries:
    queries.append(m2.group(1).strip())
  general = _query_from_text(text)
  if general and general not in queries:
    queries.append(general)
  return queries[:4]


def _shingle_jaccard(a: str, b: str, size: int = 5) -> float:
  def shingles(s: str) -> set[tuple[str, ...]]:
    words = [w.lower() for w in re.findall(r"\w+", s)]
    if len(words) < size:
      return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}

  sa, sb = shingles(a), shingles(b)
  if not sa or not sb:
    return 0.0
  return len(sa & sb) / len(sa | sb)


async def _wiki_search_titles(client: httpx.AsyncClient, query: str, limit: int) -> list[str]:
  params = {
    "action": "query",
    "format": "json",
    "list": "search",
    "srsearch": query,
    "srlimit": str(limit),
  }
  r = await client.get(
    "https://en.wikipedia.org/w/api.php",
    params=params,
    headers={"User-Agent": _USER_AGENT},
  )
  r.raise_for_status()
  return [
    item.get("title", "")
    for item in r.json().get("query", {}).get("search", [])
    if item.get("title")
  ]


async def _wiki_fetch_extracts(client: httpx.AsyncClient, titles: list[str]) -> list[CorpusChunk]:
  if not titles:
    return []
  params = {
    "action": "query",
    "format": "json",
    "prop": "extracts",
    "explaintext": "1",
    "exintro": "0",
    "exchars": "2000",
    "titles": "|".join(titles[:5]),
  }
  r = await client.get(
    "https://en.wikipedia.org/w/api.php",
    params=params,
    headers={"User-Agent": _USER_AGENT},
  )
  r.raise_for_status()
  chunks: list[CorpusChunk] = []
  for page in r.json().get("query", {}).get("pages", {}).values():
    extract = re.sub(r"\s+", " ", (page.get("extract") or "").strip())
    title = page.get("title") or "Wikipedia"
    if len(extract) < 40:
      continue
    for part in re.split(r"(?<=[.!?])\s+", extract):
      part = part.strip()
      if len(part) >= 40:
        chunks.append(CorpusChunk(
          text=part[:900],
          source="wikipedia",
          title=title,
          url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        ))
    if not chunks or chunks[-1].title != title:
      chunks.append(CorpusChunk(
        text=extract[:900],
        source="wikipedia",
        title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
      ))
  return chunks


async def fetch_wikipedia(query: str, limit: int = 5) -> list[CorpusChunk]:
  """Search Wikipedia and load full article extracts (not just short snippets)."""
  q = re.sub(r"\s+", " ", query.strip())[:120]
  if len(q) < 3:
    return []
  try:
    async with httpx.AsyncClient(timeout=18.0) as client:
      titles: list[str] = []
      seen: set[str] = set()
      for topic in _topic_queries(q) if q == query else [q]:
        for t in await _wiki_search_titles(client, topic, limit):
          if t not in seen:
            seen.add(t)
            titles.append(t)
      if not titles and q != query:
        for t in await _wiki_search_titles(client, q, limit):
          if t not in seen:
            seen.add(t)
            titles.append(t)
      return await _wiki_fetch_extracts(client, titles)
  except httpx.HTTPError:
    return []


async def fetch_wikipedia_for_text(text: str, limit: int = 4) -> list[CorpusChunk]:
  """Run multiple Wikipedia lookups from topic + keyword queries."""
  chunks: list[CorpusChunk] = []
  seen: set[str] = set()
  for q in _topic_queries(text):
    for c in await fetch_wikipedia(q, limit=limit):
      key = c.text[:60].lower()
      if key not in seen:
        seen.add(key)
        chunks.append(c)
  if not chunks:
    for c in await fetch_wikipedia(_query_from_text(text), limit=limit):
      key = c.text[:60].lower()
      if key not in seen:
        seen.add(key)
        chunks.append(c)
  return chunks


async def fetch_blogs_web(query: str, base_url: str, limit: int = 6) -> list[CorpusChunk]:
  """Blog / web snippets via free SearXNG (no paid search API)."""
  if not base_url.strip():
    return []
  base = base_url.rstrip("/")
  try:
    async with httpx.AsyncClient(timeout=14.0, follow_redirects=True) as client:
      r = await client.get(
        f"{base}/search",
        params={"q": query, "format": "json", "language": "en"},
        headers={"User-Agent": _USER_AGENT},
      )
      r.raise_for_status()
      data = r.json()
  except httpx.HTTPError:
    return []

  chunks: list[CorpusChunk] = []
  for item in (data.get("results") or [])[:limit]:
    content = (item.get("content") or item.get("snippet") or "").strip()
    if len(content) < 25:
      continue
    chunks.append(CorpusChunk(
      text=content[:600],
      source="blog",
      title=(item.get("title") or "Web article")[:200],
      url=(item.get("url") or "")[:500],
    ))
  return chunks


# ── 6–7. Similarity engine + match ranking ───────────────────────────────────

_SOURCE_LABELS = {
  "wikipedia": "Wikipedia match",
  "blog": "Blog / web match",
  "searxng": "Blog / web match",
  "pdf": "PDF document match",
  "corpus": "Local corpus match",
  "common_crawl": "Web corpus match",
}


def _best_match_for_chunk(
  chunk: str,
  vec: np.ndarray,
  live_corpus: list[CorpusChunk],
  live_vecs: np.ndarray,
) -> SentenceMatch | None:
  best_score = 0.0
  best_meta: dict[str, str] = {}

  for score, meta in search_faiss(vec, top_k=5):
    if score > best_score:
      best_score = score
      best_meta = meta

  if live_vecs.shape[0] > 0:
    sims = live_vecs @ vec
    j = int(np.argmax(sims))
    if float(sims[j]) > best_score:
      best_score = float(sims[j])
      c = live_corpus[j]
      best_meta = {"text": c.text, "source": c.source, "title": c.title, "url": c.url}

  for c in live_corpus:
    jacc = _shingle_jaccard(chunk, c.text, size=5)
    if jacc > best_score:
      best_score = jacc
      best_meta = {"text": c.text, "source": c.source, "title": c.title, "url": c.url}

  if best_score < _MATCH_THRESHOLD or not best_meta:
    return None

  src = best_meta.get("source", "corpus")
  return SentenceMatch(
    sentence="",
    sentence_index=-1,
    score=best_score,
    match_percent=int(round(min(99, best_score * 100))),
    source=f"{best_meta.get('title', src)} ({src})",
    source_type=src,
    title=best_meta.get("title", ""),
    url=best_meta.get("url", ""),
    matched_excerpt=(best_meta.get("text") or "")[:220],
    label=_SOURCE_LABELS.get(src, "Similar content detected"),
  )


def rank_matches(matches: list[SentenceMatch]) -> list[SentenceMatch]:
  return sorted(matches, key=lambda m: m.score, reverse=True)


def _sentence_index_for_chunk(chunk: str, sentences: list[str]) -> int:
  if not sentences:
    return 0
  needle = chunk[:40].lower()
  for i, s in enumerate(sentences):
    if needle and needle in s.lower():
      return i
  return 0


# ── 8. Highlighted sentences ─────────────────────────────────────────────────

def build_highlighted_sentences(
  sentences: list[str],
  ranked: list[SentenceMatch],
) -> list[dict[str, Any]]:
  highlights: list[dict[str, Any]] = []
  for m in ranked:
    if m.sentence_index < 0 or m.sentence_index >= len(sentences):
      continue
    sent = sentences[m.sentence_index]
    highlights.append({
      "sentence_index": m.sentence_index,
      "sentence": sent,
      "highlight": sent,
      "match_percent": m.match_percent,
      "risk": "high" if m.score >= _HIGH_MATCH else "medium",
      "source": m.source,
      "source_type": m.source_type,
      "url": m.url,
      "matched_excerpt": m.matched_excerpt,
      "label": m.label,
    })
  return highlights


def matches_to_segments(matches: list[SentenceMatch]) -> list[dict[str, Any]]:
  return [
    {
      "type": "embedding_match",
      "label": m.label,
      "text": m.sentence[:200],
      "match_percent": m.match_percent,
      "source": m.source,
      "matched_excerpt": m.matched_excerpt,
      "url": m.url,
      "sentence_index": m.sentence_index,
      "source_type": m.source_type,
    }
    for m in matches
  ]


# ── 9. Similarity report ─────────────────────────────────────────────────────

def build_similarity_report(
  *,
  similarity_percent: int,
  original_percent: int,
  sentences: list[str],
  ranked: list[SentenceMatch],
  sources_used: list[str],
  chunks_scanned: int,
  chunks_matched: int,
  avg_score: float,
) -> dict[str, Any]:
  high = sum(1 for m in ranked if m.score >= _HIGH_MATCH)
  medium = sum(1 for m in ranked if _MATCH_THRESHOLD <= m.score < _HIGH_MATCH)
  risk = "low" if similarity_percent < 20 else "medium" if similarity_percent < 50 else "high"
  return {
    "similarity_percent": similarity_percent,
    "original_percent": original_percent,
    "risk_level": risk,
    "sentence_count": len(sentences),
    "chunks_scanned": chunks_scanned,
    "chunks_matched": chunks_matched,
    "high_confidence_matches": high,
    "medium_confidence_matches": medium,
    "avg_embedding_similarity": round(avg_score, 3),
    "sources_checked": sources_used,
    "verdict": (
      "Likely original" if similarity_percent < 12
      else "Review recommended" if similarity_percent < 35
      else "Significant similarity — rewrite suggested"
    ),
  }


# ── Full pipeline ────────────────────────────────────────────────────────────

async def run_pipeline(
  text: str,
  *,
  searxng_url: str = "",
  use_live_wikipedia: bool = True,
  use_searxng: bool = True,
) -> PipelineResult:
  workflow: list[dict[str, str]] = []

  def step(name: str, status: str, detail: str = "") -> None:
    workflow.append({"step": name, "status": status, "detail": detail})

  try:
    _require_st()
    _require_faiss()
  except ImportError as exc:
    step("embedding_generator", "error", str(exc))
    return PipelineResult(
      workflow=workflow,
      preprocessed_text="",
      sentences=[],
      highlighted_sentences=[],
      matched_segments=[],
      similarity_percent=0,
      original_percent=100,
      sources_used=[],
      avg_embedding_similarity=0.0,
      chunks_scanned=0,
      chunks_matched=0,
      available=False,
      error=str(exc),
    )

  step("user_text", "done", f"{len(text)} chars received")

  clean = preprocess_text(text)
  step("text_preprocessing", "done", f"{len(clean)} chars after normalize")

  sentences = split_sentences(clean)
  chunks = chunk_sentences(sentences)
  step("sentence_splitting", "done", f"{len(sentences)} sentences · {len(chunks)} chunks")

  if not chunks:
    step("embedding_generator", "skip", "no chunks")
    return PipelineResult(
      workflow=workflow,
      preprocessed_text=clean,
      sentences=sentences,
      highlighted_sentences=[],
      matched_segments=[],
      similarity_percent=0,
      original_percent=100,
      sources_used=[],
      avg_embedding_similarity=0.0,
      chunks_scanned=0,
      chunks_matched=0,
      available=True,
    )

  input_vecs = generate_embeddings(chunks)
  step("embedding_generator", "done", f"model={_MODEL_NAME} · dim={input_vecs.shape[1]}")

  sources_used: list[str] = []
  if _load_index():
    pdf_hits = sum(1 for m in _meta if m.get("source") == "pdf")
    corpus_hits = len(_meta)
    step(
      "vector_database",
      "done",
      f"FAISS · {corpus_hits} vectors"
      + (f" · {pdf_hits} from PDFs" if pdf_hits else ""),
    )
    sources_used.append("faiss_index")
  else:
    step("vector_database", "warn", "index missing — run scripts/build_plagiarism_index.py")

  query = _query_from_text(clean)
  live_corpus: list[CorpusChunk] = []

  if use_live_wikipedia:
    wiki = await fetch_wikipedia_for_text(clean)
    if wiki:
      live_corpus.extend(wiki)
      sources_used.append("wikipedia")
    else:
      step("external_sources", "warn", "Wikipedia unreachable — check internet")
  if use_searxng and searxng_url:
    blogs = await fetch_blogs_web(query, searxng_url)
    if blogs:
      live_corpus.extend(blogs)
      sources_used.append("blogs_web")

  live_texts = [c.text for c in live_corpus]
  live_vecs = generate_embeddings(live_texts) if live_texts else np.zeros((0, 384), dtype=np.float32)
  if live_corpus:
    step(
      "external_sources",
      "done",
      f"Wikipedia + blogs/web · {len(live_corpus)} reference passages",
    )
  elif not sources_used:
    step("external_sources", "warn", "no live sources — build FAISS index or enable Wikipedia")

  raw_matches: list[SentenceMatch] = []
  chunk_scores: list[float] = []

  for i, (chunk, vec) in enumerate(zip(chunks, input_vecs)):
    hit = _best_match_for_chunk(chunk, vec, live_corpus, live_vecs)
    score = hit.score if hit else 0.0
    chunk_scores.append(score)
    if hit:
      sent_idx = _sentence_index_for_chunk(chunk, sentences)
      hit.sentence = chunk
      hit.sentence_index = sent_idx
      raw_matches.append(hit)

  step("similarity_engine", "done", f"compared {len(chunks)} chunks")

  ranked = rank_matches(raw_matches)
  step("match_ranking", "done", f"{len(ranked)} matches above threshold")

  highlighted = build_highlighted_sentences(sentences, ranked)
  segments = matches_to_segments(ranked)
  step("highlighted_sentences", "done", f"{len(highlighted)} sentences flagged")

  if not chunk_scores:
    avg_top = 0.0
    similarity_percent = 0
  else:
    avg_top = float(np.mean(chunk_scores))
    high = sum(1 for s in chunk_scores if s >= _HIGH_MATCH)
    medium = sum(1 for s in chunk_scores if _MATCH_THRESHOLD <= s < _HIGH_MATCH)
    n = len(chunk_scores)
    similarity_percent = int(round(min(
      98,
      max(0, (high / n) * 70 + (medium / n) * 35 + avg_top * 25),
    )))

  original_percent = 100 - similarity_percent
  report = build_similarity_report(
    similarity_percent=similarity_percent,
    original_percent=original_percent,
    sentences=sentences,
    ranked=ranked,
    sources_used=sources_used,
    chunks_scanned=len(chunks),
    chunks_matched=len(ranked),
    avg_score=avg_top,
  )
  step("similarity_report", "done", report["verdict"])

  scan_incomplete = not sources_used and similarity_percent == 0

  return PipelineResult(
    workflow=workflow,
    preprocessed_text=clean,
    sentences=sentences,
    highlighted_sentences=highlighted,
    matched_segments=segments[:20],
    similarity_percent=similarity_percent,
    original_percent=original_percent,
    sources_used=sources_used,
    avg_embedding_similarity=avg_top,
    chunks_scanned=len(chunks),
    chunks_matched=len(ranked),
    available=True,
    scan_incomplete=scan_incomplete,
    error="",
  )


# Backward-compatible entry point
async def scan_content(
  text: str,
  *,
  searxng_url: str = "",
  use_live_wikipedia: bool = True,
  use_searxng: bool = True,
) -> dict[str, Any]:
  result = await run_pipeline(
    text,
    searxng_url=searxng_url,
    use_live_wikipedia=use_live_wikipedia,
    use_searxng=use_searxng,
  )
  if not result.available:
    return {
      "available": False,
      "error": result.error,
      "similarity_percent": 0,
      "matched_segments": [],
      "sources_used": [],
      "workflow": result.workflow,
    }
  return {
    "available": True,
    "similarity_percent": result.similarity_percent,
    "original_percent": result.original_percent,
    "avg_embedding_similarity": round(result.avg_embedding_similarity, 3),
    "chunks_scanned": result.chunks_scanned,
    "chunks_matched": result.chunks_matched,
    "matched_segments": result.matched_segments,
    "highlighted_sentences": result.highlighted_sentences,
    "sources_used": result.sources_used,
    "workflow": result.workflow,
    "scan_incomplete": result.scan_incomplete,
    "similarity_report": build_similarity_report(
      similarity_percent=result.similarity_percent,
      original_percent=result.original_percent,
      sentences=result.sentences,
      ranked=[],
      sources_used=result.sources_used,
      chunks_scanned=result.chunks_scanned,
      chunks_matched=result.chunks_matched,
      avg_score=result.avg_embedding_similarity,
    ),
  }


def _chunks(text: str, *, max_len: int = 280) -> list[str]:
  return chunk_sentences(split_sentences(preprocess_text(text)), max_len=max_len)

#!/usr/bin/env python3
"""Build FAISS plagiarism index from free sources (no paid APIs).

Sources:
  - Project knowledge.jsonl + corpus.txt
  - Wikipedia API samples (per topic)
  - Common Crawl via HF allenai/c4 streaming sample (optional)

Usage:
  pip install sentence-transformers faiss-cpu httpx
  python scripts/build_plagiarism_index.py
  python scripts/build_plagiarism_index.py --wikipedia 30 --c4 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from app.engine.plagiarism_engine import CorpusChunk, build_index, configure

KNOWLEDGE = PROJECT_ROOT / "data" / "knowledge.jsonl"
CORPUS = PROJECT_ROOT / "data" / "corpus.txt"
PUBLIC_CORPUS = PROJECT_ROOT / "data" / "public_datasets_corpus.txt"
INDEX_DIR = PROJECT_ROOT / "data" / "plagiarism"

_USER_AGENT = "NexusPlagiarismChecker/1.0"
_CHUNK = 320


def _clip(text: str, n: int = _CHUNK) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= n else t[: n - 3].rstrip() + "..."


def from_knowledge() -> list[CorpusChunk]:
  chunks: list[CorpusChunk] = []
  if not KNOWLEDGE.exists():
    return chunks
  with KNOWLEDGE.open(encoding="utf-8") as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      a = _clip(row.get("a") or row.get("answer") or "")
      if len(a) < 40:
        continue
      chunks.append(CorpusChunk(
        text=a,
        source="corpus",
        title=row.get("q") or row.get("question") or "Knowledge base",
        url="",
      ))
  return chunks


def from_corpus_file(path: Path, source: str) -> list[CorpusChunk]:
  if not path.exists():
    return []
  text = path.read_text(encoding="utf-8", errors="ignore")
  parts = re.split(r"\n{2,}", text)
  return [
    CorpusChunk(text=_clip(p), source=source, title=f"{source} excerpt", url="")
    for p in parts if len(p.strip()) > 50
  ][:500]


def fetch_wikipedia(topics: list[str], per_topic: int = 2) -> list[CorpusChunk]:
  chunks: list[CorpusChunk] = []
  url = "https://en.wikipedia.org/w/api.php"
  with httpx.Client(timeout=20.0, headers={"User-Agent": _USER_AGENT}) as client:
    for topic in topics:
      try:
        r = client.get(url, params={
          "action": "query", "format": "json", "prop": "extracts",
          "explaintext": "1", "exintro": "0", "exchars": "800",
          "generator": "search", "gsrsearch": topic, "gsrlimit": str(per_topic),
        })
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
          extract = _clip(page.get("extract") or "", 500)
          title = page.get("title") or topic
          if len(extract) < 50:
            continue
          chunks.append(CorpusChunk(
            text=extract,
            source="wikipedia",
            title=title,
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
          ))
      except httpx.HTTPError:
        continue
  return chunks


def sample_c4(limit: int) -> list[CorpusChunk]:
  try:
    from datasets import load_dataset
  except ImportError:
    print("Skip C4: pip install datasets")
    return []
  chunks: list[CorpusChunk] = []
  ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
  for row in ds:
    text = _clip(str(row.get("text") or ""), 500)
    if len(text) < 80:
      continue
    chunks.append(CorpusChunk(
      text=text,
      source="common_crawl",
      title="C4 web sample",
      url=str(row.get("url") or ""),
    ))
    if len(chunks) >= limit:
      break
  return chunks


def from_pdf_dir(pdf_dir: Path) -> list[CorpusChunk]:
  """Extract text from PDF files (free — requires pypdf)."""
  if not pdf_dir.is_dir():
    return []
  try:
    from pypdf import PdfReader
  except ImportError:
    print("Skip PDFs: pip install pypdf")
    return []

  chunks: list[CorpusChunk] = []
  for pdf in sorted(pdf_dir.glob("*.pdf")):
    try:
      reader = PdfReader(str(pdf))
      text_parts: list[str] = []
      for page in reader.pages[:30]:
        text_parts.append(page.extract_text() or "")
      full = _clip(" ".join(text_parts), 800)
      if len(full) < 50:
        continue
      chunks.append(CorpusChunk(
        text=full,
        source="pdf",
        title=pdf.stem,
        url=str(pdf.resolve()),
      ))
    except Exception as exc:
      print(f"Skip PDF {pdf.name}: {exc}")
  return chunks


def main() -> None:
  parser = argparse.ArgumentParser(description="Build plagiarism FAISS index")
  parser.add_argument("--wikipedia", type=int, default=25, help="Wikipedia topic searches")
  parser.add_argument("--c4", type=int, default=40, help="Common Crawl C4 samples (0=skip)")
  parser.add_argument("--pdf-dir", type=str, default="", help="Folder of PDF files to index")
  parser.add_argument("--out", type=str, default=str(INDEX_DIR))
  args = parser.parse_args()

  configure(Path(args.out))
  all_chunks: list[CorpusChunk] = []

  all_chunks.extend(from_knowledge())
  all_chunks.extend(from_corpus_file(CORPUS, "corpus"))
  all_chunks.extend(from_corpus_file(PUBLIC_CORPUS, "common_crawl"))

  topics: list[str] = []
  if KNOWLEDGE.exists():
    with KNOWLEDGE.open(encoding="utf-8") as f:
      for line in f:
        row = json.loads(line)
        q = (row.get("q") or row.get("question") or "").strip()
        if q and len(q) > 5:
          topics.append(q[:80])
  topics = list(dict.fromkeys(topics))[: args.wikipedia]
  if not topics:
    topics = ["technology", "business", "health", "science", "education", "marketing", "law"]
  all_chunks.extend(fetch_wikipedia(topics, per_topic=1))

  if args.c4 > 0:
    all_chunks.extend(sample_c4(args.c4))

  pdf_dir = Path(args.pdf_dir) if args.pdf_dir else INDEX_DIR / "pdfs"
  if pdf_dir.is_dir():
    pdf_chunks = from_pdf_dir(pdf_dir)
    all_chunks.extend(pdf_chunks)
    print(f"PDFs: {len(pdf_chunks)} chunks from {pdf_dir}")

  seen: set[str] = set()
  unique: list[CorpusChunk] = []
  for c in all_chunks:
    key = c.text[:80].lower()
    if key in seen:
      continue
    seen.add(key)
    unique.append(c)

  print(f"Building index with {len(unique)} chunks…")
  n = build_index(unique)
  print(f"Done — {n} vectors at {args.out}")


if __name__ == "__main__":
  main()

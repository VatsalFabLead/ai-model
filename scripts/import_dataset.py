#!/usr/bin/env python3
"""Import free open datasets into the knowledge base (data/knowledge.jsonl).

Supported inputs (no third-party models involved):
  - .jsonl  with fields q/a (or question/answer, title/text)
  - .csv    with columns q,a (or question,answer, title,text)
  - .txt    blocks separated by blank lines: line 1 = question, rest = answer
            or "Q: ... / A: ..." pairs

Usage:
  python scripts/import_dataset.py path/to/file1.csv path/to/file2.jsonl
  python scripts/import_dataset.py data/raw/         (imports a whole folder)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge.jsonl"


def _pick(d: dict, *keys: str) -> str:
  for k in keys:
    if d.get(k):
      return str(d[k]).strip()
  return ""


def from_jsonl(path: Path) -> list[dict]:
  out = []
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
    except json.JSONDecodeError:
      continue
    a = _pick(obj, "a", "answer", "text")
    q = _pick(obj, "q", "question", "title") or a
    if a:
      out.append({"q": q, "a": a})
  return out


def from_csv(path: Path) -> list[dict]:
  out = []
  with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      low = {k.lower().strip(): v for k, v in row.items() if k}
      a = _pick(low, "a", "answer", "text")
      q = _pick(low, "q", "question", "title") or a
      if a:
        out.append({"q": q, "a": a})
  return out


def from_txt(path: Path) -> list[dict]:
  out = []
  text = path.read_text(encoding="utf-8")
  blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
  for block in blocks:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
      continue
    if lines[0].lower().startswith("q:"):
      q = lines[0][2:].strip()
      a = " ".join(ln[2:].strip() if ln.lower().startswith("a:") else ln for ln in lines[1:])
      a = a.strip()
    else:
      q = lines[0]
      a = " ".join(lines[1:]).strip() or lines[0]
    if a:
      out.append({"q": q, "a": a})
  return out


def collect_files(paths: list[str]) -> list[Path]:
  files: list[Path] = []
  for p in paths:
    path = Path(p)
    if path.is_dir():
      files.extend(sorted(path.rglob("*.jsonl")))
      files.extend(sorted(path.rglob("*.csv")))
      files.extend(sorted(path.rglob("*.txt")))
    elif path.exists():
      files.append(path)
    else:
      print(f"Skipping missing path: {p}")
  return files


def main() -> None:
  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

  files = collect_files(sys.argv[1:])
  if not files:
    print("No input files found.")
    sys.exit(1)

  existing: set[str] = set()
  if KNOWLEDGE_FILE.exists():
    for line in KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        existing.add(json.loads(line).get("q", "").strip().lower())
      except json.JSONDecodeError:
        continue

  new_entries: list[dict] = []
  for f in files:
    if f.suffix == ".jsonl":
      entries = from_jsonl(f)
    elif f.suffix == ".csv":
      entries = from_csv(f)
    elif f.suffix == ".txt":
      entries = from_txt(f)
    else:
      continue
    for e in entries:
      key = e["q"].strip().lower()
      if key and key not in existing:
        existing.add(key)
        new_entries.append(e)
    print(f"{f}: parsed {len(entries)} entries")

  KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
  with KNOWLEDGE_FILE.open("a", encoding="utf-8") as f:
    for e in new_entries:
      f.write(json.dumps(e, ensure_ascii=False) + "\n")

  print(f"Added {len(new_entries)} new entries. Total knowledge file: {KNOWLEDGE_FILE}")


if __name__ == "__main__":
  main()

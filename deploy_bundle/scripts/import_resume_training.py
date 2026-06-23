#!/usr/bin/env python3
"""Merge resume training knowledge into the main knowledge base.

Resume builder uses data/resume_knowledge.jsonl directly. This script also
copies entries into data/knowledge.jsonl for chat RAG — no GPT/Claude/Gemini.

Usage:
  python scripts/import_resume_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESUME_KB = PROJECT_ROOT / "data" / "resume_knowledge.jsonl"
MAIN_KB = PROJECT_ROOT / "data" / "knowledge.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
  rows = []
  if not path.exists():
    return rows
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return rows


def main() -> None:
  schema_rows = _load_jsonl(RESUME_KB)
  if not schema_rows:
    print(f"No entries in {RESUME_KB}")
    sys.exit(1)

  existing = _load_jsonl(MAIN_KB)
  existing_q = {(r.get("q") or "").strip().lower() for r in existing}
  to_add = [r for r in schema_rows if (r.get("q") or "").strip().lower() not in existing_q]

  print(f"Resume KB entries: {len(schema_rows)}")
  print(f"New entries to add: {len(to_add)}")

  if not to_add:
    print("Nothing new to import.")
    return

  with MAIN_KB.open("a", encoding="utf-8") as f:
    for row in to_add:
      f.write(json.dumps({"q": row.get("q", ""), "a": row.get("a", "")}, ensure_ascii=False) + "\n")

  print(f"Appended {len(to_add)} entries to {MAIN_KB}")


if __name__ == "__main__":
  main()

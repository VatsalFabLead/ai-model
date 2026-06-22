#!/usr/bin/env python3
"""Merge schema training knowledge into the main knowledge base.

Schema markup uses data/schema_knowledge.jsonl directly. This script also
copies those entries into data/knowledge.jsonl so your custom chat model
can answer schema questions via RAG — no GPT/Claude/Gemini involved.

Usage:
  python scripts/import_schema_training.py
  python scripts/import_schema_training.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_KB = PROJECT_ROOT / "data" / "schema_knowledge.jsonl"
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
  parser = argparse.ArgumentParser(description="Import schema training into knowledge.jsonl")
  parser.add_argument("--dry-run", action="store_true", help="Show count only, do not write")
  args = parser.parse_args()

  schema_rows = _load_jsonl(SCHEMA_KB)
  if not schema_rows:
    print(f"No entries in {SCHEMA_KB}")
    sys.exit(1)

  existing = _load_jsonl(MAIN_KB)
  existing_q = {(r.get("q") or "").strip().lower() for r in existing}
  to_add = [r for r in schema_rows if (r.get("q") or "").strip().lower() not in existing_q]

  print(f"Schema KB entries: {len(schema_rows)}")
  print(f"New entries to add: {len(to_add)}")

  if args.dry_run:
    return

  if not to_add:
    print("Nothing new to import.")
    return

  with MAIN_KB.open("a", encoding="utf-8") as f:
    for row in to_add:
      f.write(json.dumps({"q": row.get("q", ""), "a": row.get("a", "")}, ensure_ascii=False) + "\n")

  print(f"Appended {len(to_add)} entries to {MAIN_KB}")
  print("Restart the server to reload knowledge.")


if __name__ == "__main__":
  main()

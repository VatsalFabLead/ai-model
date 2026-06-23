#!/usr/bin/env python3
"""Build worldwide chat knowledge + corpus (100% custom, no GPT/Claude/Gemini).

Includes Nexus capabilities, rules, personality, categories, and languages.

Usage:
  python scripts/import_chat_training.py
  python scripts/import_chat_training.py --build-corpus
  python scripts/import_chat_training.py --build-corpus --epochs 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.assistant_profile import knowledge_entries

CHAT_KB = PROJECT_ROOT / "data" / "chat_knowledge.jsonl"
MAIN_KB = PROJECT_ROOT / "data" / "knowledge.jsonl"
MAIN_CORPUS = PROJECT_ROOT / "data" / "corpus.txt"
PROFILE_CORPUS = PROJECT_ROOT / "data" / "assistant_profile_corpus.txt"


def corpus_pairs(entries: list[dict[str, str]]) -> str:
  lines: list[str] = []
  for row in entries:
    lines.append(f"User: {row['q']}\nAssistant: {row['a']} <|eos|>\n")
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description="Import Nexus assistant training data")
  parser.add_argument("--build-corpus", action="store_true")
  parser.add_argument("--epochs", type=int, default=0)
  args = parser.parse_args()

  entries = knowledge_entries()
  CHAT_KB.write_text(
    "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote {len(entries)} profile entries to {CHAT_KB}")

  existing_q: set[str] = set()
  if MAIN_KB.exists():
    for line in MAIN_KB.read_text(encoding="utf-8").splitlines():
      if line.strip():
        try:
          existing_q.add(json.loads(line).get("q", "").strip().lower())
        except json.JSONDecodeError:
          pass

  to_add = [e for e in entries if e["q"].lower() not in existing_q]
  if to_add:
    with MAIN_KB.open("a", encoding="utf-8") as f:
      for row in to_add:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended {len(to_add)} new entries to {MAIN_KB}")
  else:
    print("No new KB entries (already imported)")

  text = corpus_pairs(entries)
  PROFILE_CORPUS.write_text(text, encoding="utf-8")
  print(f"Wrote profile corpus: {PROFILE_CORPUS}")

  if args.build_corpus:
    with MAIN_CORPUS.open("a", encoding="utf-8") as f:
      f.write("\n" + text)
    print(f"Merged profile corpus into {MAIN_CORPUS}")

  if args.epochs > 0:
    cmd = [
      sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
      "--corpus", str(MAIN_CORPUS), "--epochs", str(args.epochs), "--seq-len", "128",
    ]
    print("Training custom model:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

  print("Restart server to reload knowledge and system prompt.")


if __name__ == "__main__":
  main()
